from pathlib import Path
import time
from typing import Any

from loguru import logger
from opik import opik_context, track
from smolagents import MultiStepAgent, ToolCallingAgent, MessageRole, LiteLLMModel
from smolagents.memory import ActionStep

from second_brain_online.config import settings
from second_brain_online.infrastructure.cache.cache_manager import CacheManager
from .tools import (
    MongoDBRetrieverTool, HuggingFaceSummarizerTool, OpenAISummarizerTool, what_can_i_do, GeminiSummarizerTool
)
from second_brain_online.exceptions import LLMGenerationError, LLMQuotaError, AgentError


def get_agent(retriever_config_path: Path) -> "AgentWrapper":
    agent = AgentWrapper.build_from_smolagents(
        retriever_config_path=retriever_config_path
    )
    return agent


class AgentWrapper:

    def __init__(self, agent: MultiStepAgent) -> None:
        self.__agent = agent
        # Cache lives here — caches FINAL ANSWER, not retriever chunks
        self.__cache = CacheManager(semantic_threshold=0.95)
        # Tracks cache status per run — read in finally block to build trace metadata.
        # Reset at the start of every run() call so stale values never bleed across requests.
        self.__cache_status: str = "miss"

    @property
    def input_messages(self) -> list[dict]:
        return self.__agent.input_messages

    @property
    def agent_name(self) -> str:
        return self.__agent.agent_name

    @property
    def max_steps(self) -> str:
        return self.__agent.max_steps

    @classmethod
    def build_from_smolagents(cls, retriever_config_path: Path) -> "AgentWrapper":
        retriever_tool = MongoDBRetrieverTool(retriever_config_path)

        if settings.USE_HUGGINGFACE_DEDICATED_ENDPOINT:
            logger.warning(
                f"Using Hugging Face dedicated endpoint as the summarizer with URL: {settings.HUGGINGFACE_DEDICATED_ENDPOINT}"
            )
            summarizer_tool = HuggingFaceSummarizerTool()
        else:
            logger.warning(
                f"Using Google Gemini model as the summarizer with model: {settings.GEMINI_MODEL_ID}"
            )
            summarizer_tool = GeminiSummarizerTool()
            
        # model = LiteLLMModel(
        #     model_id=settings.OPENAI_MODEL_ID,
        #     api_base="https://api.openai.com/v1",
        #     api_key=settings.OPENAI_API_KEY,
        # )
        
        # openrouter compatible model
        
        # model = LiteLLMModel(
        #     model_id=settings.OPENROUTER_MODEL_ID, # Use the gemini/ prefix
        #     max_tokens=2048,
        #     api_key=settings.OPENROUTER_API_KEY,
        #     base_url=settings.OPENROUTER_API_BASE,
        # )

        model = LiteLLMModel(
            model_id=settings.GEMINI_MODEL_ID,
            max_tokens=2048,
            api_key=settings.GEMINI_API_KEY,
            num_retries=2,
            retry_after=3,
        )

        agent = ToolCallingAgent(
            tools=[what_can_i_do, retriever_tool, summarizer_tool],
            model=model,
            max_steps=2,
            verbosity_level=2,
            instructions="""You are a retrieval-based assistant.
                CRITICAL RULES:
                - You MUST use the mongodb_vector_search_retriever tool to answer EVERY question.
                - If the tool fails or returns no results, respond ONLY with:
                "I was unable to find relevant information in the knowledge base."
                - NEVER answer from your own knowledge or training data.
                """
        )

        return cls(agent)

    @track(name="AgentWrapper.run")
    def run(self, task: str, **kwargs) -> Any:
        # Reset cache status at the start of every request
        self.__cache_status = "miss"
        
        # ── TOTAL request time ─────────────────────────────────────────────────
        t_total_start = time.perf_counter()
        logger.info(f"[TIMING] AgentWrapper.run() started | query='{task[:80]}...'")

        try:
            # ── TIER 1: Redis exact match on final answer ──────────────────────
            cached = self.__cache.get_exact(task)
            if cached:
                t_total_end = time.perf_counter()
                logger.info(
                    f"[TIMING] TOTAL request time: {t_total_end - t_total_start:.3f}s "
                    f"| path=cache_hit_exact"
                )
                logger.debug("Final answer cache HIT [exact]")
                self.__cache_status = "hit_exact"
                return cached

            # ── TIER 2: MongoDB semantic match on final answer ─────────────────
            query_embedding = self.__cache.embed_query(task)
            cached = self.__cache.get_semantic(query_embedding)
            if cached:
                t_total_end = time.perf_counter()
                logger.info(
                    f"[TIMING] TOTAL request time: {t_total_end - t_total_start:.3f}s "
                    f"| path=cache_hit_semantic"
                )
                logger.debug("Final answer cache HIT [semantic]")
                self.__cache_status = "hit_semantic"
                self.__cache.set_exact(task, cached)
                return cached

            # ── TIER 3: Full agent run ─────────────────────────────────────────
            t_agent_start = time.perf_counter()
            result = self.__agent.run(task, **kwargs)
            t_agent_end = time.perf_counter()
            logger.info(
                f"[TIMING] agent.run() (Gemini + tools + summary): "
                f"{t_agent_end - t_agent_start:.3f}s"
            ) 
            
            result_str = str(result)  

            # Detect smolagents hard failures
            if isinstance(result, str) and "Error in generating final LLM output" in result:
                if (
                    "RateLimitError" in result_str
                    or "quota exceeded" in result_str.lower()
                    or "RESOURCE_EXHAUSTED" in result_str
                ):
                    raise LLMQuotaError("Model quota exceeded. Please try again later.")
                raise LLMGenerationError("Failed to generate response from LLM.")

            # Detect tool failure fallback — LLM answered from own knowledge
            tool_succeeded = any(
                isinstance(step, ActionStep)
                and getattr(step, "error", None) is None
                and any(
                    getattr(tc, "name", None) == "mongodb_vector_search_retriever"
                    for tc in (getattr(step, "tool_calls", None) or [])
                )
                for step in getattr(self.__agent.memory, "steps", [])
            )

            if not tool_succeeded:
                logger.warning("Retriever tool never succeeded — suppressing LLM fallback")
                return "I was unable to find relevant information in the knowledge base."

            # Store final answer in both cache tiers
            self.__cache.set_exact(task, result_str)
            self.__cache.set_semantic(query_embedding, result_str)
            
            t_total_end = time.perf_counter()
            logger.info(
                f"[TIMING] TOTAL request time: {t_total_end - t_total_start:.3f}s "
                f"| path=full_agent_run"
            )

            return result

        except AgentError:
            raise

        except Exception as e:
            logger.exception("Agent execution failed")
            raise AgentError("Agent execution failed") from e

        finally:
            # ── Opik tracing — fully safe, never breaks execution ──────────────
            # smolagents 1.10+ removed several attributes that existed in 1.4.x:
            #   - system_prompt_template → gone
            #   - tool_description_template → gone
            #   - model.last_input_token_count → gone (now tracked per-step in logs)
            #   - model.last_output_token_count → gone
            # All fields now use getattr with safe defaults.
            try:
                model = self.__agent.model

                # Extract token counts from step logs (new way in 1.10+)
                input_tokens, output_tokens = self.__extract_token_counts()

                metadata = {
                    # Cache status — the key field, correctly set for all paths
                    "cache_status": self.__cache_status,
                    "cache_hit": self.__cache_status != "miss",

                    # Agent attrs
                    "system_prompt": getattr(self.__agent, "system_prompt", None),
                    "tools": list(getattr(self.__agent, "tools", {}).keys()),
                    "max_steps": getattr(self.__agent, "max_steps", None),
                    "step_number": getattr(self.__agent, "step_number", None),

                    # Model attrs — safe for 1.4 and 1.10+
                    "model_id": getattr(model, "model_id", None),
                    "api_base": getattr(model, "api_base", None),

                    # Token counts — extracted from logs in 1.10+
                    "input_token_count": input_tokens,
                    "output_token_count": output_tokens,
                }

                opik_context.update_current_trace(
                    tags=["agent", f"cache_{self.__cache_status}"],
                    metadata=metadata,
                )

            except Exception:
                logger.warning("Failed to update opik trace metadata")

    def __extract_token_counts(self) -> tuple[int, int]:
        """
        smolagents 1.24: token counts live in agent.memory.steps.
        Returns (0, 0) for cache hits since the agent was never run.
        """
        input_tokens, output_tokens = 0, 0
        try:
            for step in getattr(self.__agent.memory, "steps", []):
                token_usage = getattr(step, "token_usage", None)
                if token_usage is not None:
                    input_tokens += getattr(token_usage, "input_tokens", 0) or 0
                    output_tokens += getattr(token_usage, "output_tokens", 0) or 0
        except Exception as e:
            logger.debug(f"Token count extraction failed: {e}")
        return input_tokens, output_tokens


def extract_tool_response(agent: ToolCallingAgent) -> str:
    """
    Extracts and concatenates all tool response contents with numbered observation delimiters.
    """
    tool_response = [
        msg["content"]
        for msg in agent.input_messages
        if msg["role"] == MessageRole.TOOL_RESPONSE
    ]

    return "\n".join(
        f"-------- OBSERVATION {i + 1} --------\n{response}"
        for i, response in enumerate(tool_response)
    )


class OpikCallbackManager:
    def __init__(self) -> None:
        self.output_state: dict = {}

    def __call__(self, step_log) -> None:
        input_state = {
            "agent_memory": step_log.agent_memory,
            "tool_calls": step_log.tool_calls
        }
        self.output_state = {"observations": step_log.observations}
        self.trace(input_state)

    @track(name="Callback.run")
    def trace(self, step_log) -> dict:
        return self.output_state