import json
from typing import Any

from pydantic import BaseModel
from opik import exceptions
from opik.evaluation.metrics import base_metric, score_result
from opik.evaluation.models import LiteLLMChatModel

from second_brain_online.config import settings

class SummaryDensityJudgeResponse(BaseModel):
    score: float
    reason: str

class SummaryDensityJudge(base_metric.BaseMetric):
    def __init__(
        self, 
        name: str = "summary_density_judge",
        model_name: str = settings.GEMINI_MODEL_ID
    ):
        self.name = name
        self.client = LiteLLMChatModel(model_name=model_name)
        self.prompt_template = """
        You are an impartial expert judge. Evaluate the quality of a given answer to an instruction based on how long the answer it is. 

How to decide whether the lengths of the answer is appropriate:
1 (Poor): Too short, does not answer the question OR too long, it contains too much noise and unrequired information, where the answer could be more concise.
2 (Good): Good lengthbalance of the answer, but the answer is still too short OR too long.
3 (Excellent): The length of the answer is appropriate, it answers the question and is not too long or too short.

Example of bad answer that is too short: 
<answer>
LangChain, LlamaIndex, Haystack
</answer>

Example of bad answer that is too long:
<answer>
LangChain is a powerful and versatile framework designed specifically for building sophisticated LLM applications. It provides comprehensive abstractions for essential components like prompting, memory management, agent behaviors, and chain orchestration. The framework boasts an impressive ecosystem with extensive integrations across various tools and services, making it highly flexible for diverse use cases. However, this extensive functionality comes with a steeper learning curve that might require dedicated time to master.

LlamaIndex (which was formerly known as GPTIndex) has carved out a specialized niche in the LLM tooling landscape, focusing primarily on data ingestion and advanced indexing capabilities for Large Language Models. It offers a rich set of sophisticated mechanisms to structure and query your data, including vector stores for semantic similarity search, keyword indices for traditional text matching, and tree indices for hierarchical data organization. While it particularly shines in Retrieval-Augmented Generation (RAG) applications, its comprehensive feature set might be excessive for more straightforward implementation needs.

Haystack stands out as a robust end-to-end framework that places particular emphasis on question-answering systems and semantic search capabilities. It provides a comprehensive suite of document processing tools and comes equipped with production-ready pipelines that can be deployed with minimal configuration. The framework includes advanced features like multi-stage retrieval, document ranking, and reader-ranker architectures. While these capabilities make it powerful for complex information retrieval tasks, new users might find the initial configuration and architecture decisions somewhat challenging to navigate.

Each of these frameworks brings unique strengths to the table while sharing some overlapping functionality. The choice between them often depends on specific use cases, technical requirements, and team expertise. LangChain offers the broadest general-purpose toolkit, LlamaIndex excels in data handling and RAG, while Haystack provides the most streamlined experience for question-answering systems.
</answer>

Example of excellent answer that is appropriate:
<answer>
1. LangChain is a powerful framework for building LLM applications that provides abstractions for prompting, memory, agents, and chains. It has extensive integrations with various tools and services, making it highly flexible but potentially complex to learn. 
2. LlamaIndex specializes in data ingestion and indexing for LLMs, offering sophisticated ways to structure and query your data through vector stores, keyword indices, and tree indices. It excels at RAG applications but may be overkill for simpler use cases. 
3. Haystack is an end-to-end framework focused on question-answering and semantic search, with strong document processing capabilities and ready-to-use pipelines. While powerful, its learning curve can be steep for beginners. 
</answer>

Instruction: {input}

Answer: {output}

Provide your evaluation in JSON format:

{{
  "score": 1 | 2 | 3,
  "reason": "..."
}}

Where:
- 1 = Poor
- 2 = Good
- 3 = Excellent
"""

    def score(self, input: str, output: str, **ignored_kwargs: Any) -> score_result.ScoreResult:
        """
        Score the output of an LLM.

        Args:
            output: The output of an LLM to score.
            **ignored_kwargs: Any additional keyword arguments. This is important so that the metric can be used in the `evaluate` function.
        """
        
        self.prompt_template.format(input=input, output=output)
        
        response = self.client.generate_string(
            input,
            response_format=SummaryDensityJudgeResponse
        )
        
        return self._parse_llm_response(response)


    def _parse_llm_reponse(self, response: SummaryDensityJudgeResponse):
        try:
            dict_content = json.loads(response)
        except:
            raise exceptions.MetricComputationError("Failed to parse llm model response.")
        
        score = dict_content["score"]
        
        try: 
            assert 1 <= score <= 3, f"invalid score value {score}"     
        except AssertionError as e:
            raise exceptions.MetricComputationError(str(e))
        
        score = (score - 1) /   2.0  # normalize to range of 0-1
        
        return score_result(
            name=self.name,
            value=score,
            reason=dict_content["reason"],
        )