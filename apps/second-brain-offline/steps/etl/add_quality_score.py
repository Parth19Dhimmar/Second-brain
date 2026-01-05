from loguru import logger
from typing import Annotated

from zenml import step, get_step_context

from second_brain_offline.domain import Document
from second_brain_offline.application.agents import QualityScoreAgent, HeuristicQualityAgent

@step
def add_quality_score(
    documents: list[Document],
    model_id: str = "gpt-4o-mini",
    mock: bool = False,
    max_workers: int = 10,
) -> Annotated[list[Document], "scored_documents"]:
    """Adds quality scores to documents using heuristic and model-based scoring agents.

    This function processes documents in two stages:
    1. Applies heuristic-based quality scoring
    2. Uses a model-based quality agent for documents that weren't scored by heuristics

    Args:
        documents: List of documents to evaluate for quality
        model_id: Identifier for the model to use in quality assessment.
            Defaults to "gpt-4o-mini"
        mock: If True, uses mock responses instead of real model calls.
            Defaults to False
        max_workers: Maximum number of concurrent quality check operations. 
            Defaults to 10

    Returns:
        list[Document]: Documents enhanced with quality scores, annotated as
            "scored_documents" for pipeline metadata tracking

    Note:
        The function adds metadata to the step context including the total number
        of documents and how many received quality scores.
    """

    heuristic_agent = HeuristicQualityAgent()
    
    scored_documents : list[Document] = heuristic_agent(documents=documents)
    
    scored_documents_with_heuristics = [
        doc for doc in scored_documents if doc.content_quality_score is not None
    ]
    
    documents_without_score = [
        doc for doc in scored_documents if doc.content_quality_score is None
    ]
    
    quality_score_agent = QualityScoreAgent(
        model_id=model_id,
        mock=False,
        concurrent_requests=max_workers
    )
    
    scored_documents_with_agent: list[Document] = quality_score_agent(documents_without_score)
    
    scored_documents: list[Document] = (
        scored_documents_with_heuristics + scored_documents_with_agent
    )
    
    len_documents = len(documents)
    len_documents_with_score = len(
        [doc for doc in scored_documents if doc.content_quality_score is not None]
    )
    
    logger.info(f"Total documents: {len_documents}")
    logger.info(f"Total documents that were scored: {len_documents_with_score}")

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="scored_documents",
        metadata={
            "len_documents": len_documents,
            "len_documents_with_scores": len_documents_with_score,
            "len_documents_scored_with_heuristics": len(
                scored_documents_with_heuristics
            ),
            "len_documents_scored_with_agents": len(scored_documents_with_agent),
        },
    )

    return scored_documents
    
    
        
    
    

    