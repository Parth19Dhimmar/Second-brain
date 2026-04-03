import time
import json
import yaml
from pathlib import Path

from opik import opik_context, track
from loguru import logger
from smolagents import Tool

from second_brain_online.application.rag import get_retriever
from second_brain_online.utilities.retries import mongo_retry


class MongoDBRetrieverTool(Tool):
    name = "mongodb_vector_search_retriever"
    description = """Use this tool to search and retrieve relevant documents from a knowledge base using semantic search.
    This tool performs similarity-based search to find the most relevant documents matching the query.
    Best used when you need to:
    - Find specific information from stored documents
    - Get context about a topic
    - Research historical data or documentation
    The tool will return multiple relevant document snippets."""
    inputs = {
        "query": {
            "type": "string",
            "description": """The search query to find relevant documents for using semantic search.
            Should be a clear, specific question or statement about the information you're looking for.""",
        }
    }
    output_type = "string"

    def __init__(self, config_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.config_path = config_path

        # Parse config once
        config = yaml.safe_load(config_path.read_text())["parameters"]
        self.retriever = self.__load_retriever(config)

    def __load_retriever(self, config: dict):
        return get_retriever(
            embedding_model_id=config["embedding_model_id"],
            embedding_model_type=config["embedding_model_type"],
            retriever_type=config["retriever_type"],
            k=5,
            device=config["device"],
        )

    def __get_retriever_meta(self) -> dict:
        return {
            "model_id": getattr(
                self.retriever.base_retriever.vectorstore._embedding,
                "model_name",
                "unknown",
            ),
            "k": getattr(self.retriever, "k", None),
        }

    def __format_docs(self, result_docs: list) -> str:
        formatted_docs = []
        for i, doc in enumerate(result_docs):
            formatted_docs.append(
                f"""
                <document id={i}>
                <title>{doc.metadata.get("title")}</title>
                <url>{doc.metadata.get("url")}</url>
                <content>{doc.page_content.strip()}</content>
                </document>
                """
            )

        result = "\n".join(formatted_docs)
        return f"""
            <search_results>
            {result}
            </search_results>
            When using context from any document, also include the document URL as reference, which is found in the <url> tag.
        """

    @mongo_retry
    def __invoke_retriever(self, query: str):
        """
        Isolated so mongo_retry only wraps the Atlas network call,
        not the entire forward() method with its opik tracing and formatting.
        Only ServerSelectionTimeoutError / NetworkTimeout / AutoReconnect trigger retry.
        Quota errors, auth errors, bad queries — fail immediately.
        """
        return self.retriever.invoke(query)

    @track(name="MongoDBRetrieverTool.forward")
    def forward(self, query: str) -> str:
        # ── TOTAL tool time ───────────────────────────────────────────────────
        t_tool_start = time.perf_counter()
        
        parsed_query = self.__parse_query(query)

        opik_context.update_current_trace(
            tags=["agent"],
            metadata=self.__get_retriever_meta(),
        )

        try:
            # result_docs = self.__invoke_retriever(parsed_query)
            # return self.__format_docs(result_docs)
            # ── 1. Query embedding (gte-large on CPU — expected bottleneck) ───
            # The retriever embeds the query internally inside invoke().
            # We time the full invoke() then break it down further below.
            # If this is slow, the embedding step inside is the cause.
            t_invoke_start = time.perf_counter()
            result_docs = self.__invoke_retriever(parsed_query)
            t_invoke_end = time.perf_counter()
 
            logger.info(
                f"[TIMING] retriever.invoke() total: {t_invoke_end - t_invoke_start:.3f}s "
                f"| docs returned: {len(result_docs)} "
                f"| query: '{parsed_query[:60]}...'"
            )
 
            # ── 2. Format docs ────────────────────────────────────────────────
            t_fmt_start = time.perf_counter()
            formatted = self.__format_docs(result_docs)
            t_fmt_end = time.perf_counter()
            logger.debug(f"[TIMING] format_docs: {t_fmt_end - t_fmt_start:.3f}s")
 
            t_tool_end = time.perf_counter()
            logger.info(
                f"[TIMING] MongoDBRetrieverTool.forward() TOTAL: {t_tool_end - t_tool_start:.3f}s"
            )
 
            return formatted


        except Exception:
            logger.opt(exception=True).debug("Error retrieving documents after retries")
            return "Error retrieving documents."

    @track(name="MongoDBRetrieverTool.parse_query")
    def __parse_query(self, query: str) -> str:
        """
        Handles two formats depending on which LLM/agent calls the tool:
          - Plain string:  "What is attention mechanism?"       <- smolagents + Gemini
          - JSON-wrapped:  {"query": "What is attention..."}    <- older OpenAI-style agents
        """
        query = query.strip()
        if query.startswith("{"):
            try:
                return json.loads(query)["query"]
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse query as JSON, falling back to raw string")
        return query