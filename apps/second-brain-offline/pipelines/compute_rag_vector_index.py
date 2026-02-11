from zenml import pipeline
from loguru import logger

logger.info("🔥 compute_rag_vector_index module imported")

from second_brain_offline.application.rag import EmbeddingModelType
from second_brain_offline.application.rag import RetrieverType
from second_brain_offline.application.rag import SummarizationType
from steps.compute_rag_vector_index import chunk_embed_load_async, filter_by_quality  # CHANGED: chunk_embed_load -> chunk_embed_load_async
from steps.infrastructure import fetch_from_mongodb


@pipeline(enable_cache=False)
def compute_rag_vector_index(
    extract_collection_name: str,
    fetch_limit: int,
    load_collection_name: str,
    content_quality_score_threshold: float,
    retriever_type: RetrieverType,
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType,
    embedding_model_dim: int,
    chunk_size: int,
    contextual_summarization_type: SummarizationType = "none",
    contextual_agent_model_id: str | None = None,
    contextual_agent_max_characters: int | None = None,
    mock: bool = False,
    processing_batch_size: int = 32,                    # CHANGED: Default 256 -> 32 (better memory management)
    processing_max_concurrent: int = 50,                # CHANGED: processing_max_workers -> processing_max_concurrent, default 10 -> 50
    device: str = "cpu",
    enable_retry: bool = True,                          # NEW: Enable retry logic
    max_retries: int = 3,                               # NEW: Max retry attempts
) -> None:

    documents = fetch_from_mongodb(
        collection_name=extract_collection_name, 
        limit=fetch_limit
    )
    
    logger.info(f"threshold : {content_quality_score_threshold}")
    
    documents = filter_by_quality(
        documents=documents,
        content_quality_score_threshold=content_quality_score_threshold,
    )
    
    chunk_embed_load_async(                             # CHANGED: chunk_embed_load -> chunk_embed_load_async
        documents=documents,
        embed_collection_name=load_collection_name,
        retriever_type=retriever_type,
        contextual_summarization_type=contextual_summarization_type,
        contextual_agent_model_id=contextual_agent_model_id,
        contextual_agent_max_characters=contextual_agent_max_characters,
        embedding_model_type=embedding_model_type,
        embedding_model_id=embedding_model_id,
        embedding_model_dim=embedding_model_dim,
        processing_max_concurrent=processing_max_concurrent,  # CHANGED: processing_max_workers -> processing_max_concurrent
        processing_batch_size=processing_batch_size,
        mock=mock,
        chunk_size=chunk_size,
        device=device,
        enable_retry=enable_retry,                      # NEW
        max_retries=max_retries,                        # NEW
    )

if __name__ == "__main__":
    logger.info("🚀 compute_rag_vector_index __main__ started")
    import yaml
    
    with open("configs/compute_rag_vector_index_huggingface_contextual_none.yaml", "r") as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded configurations: {config}")
    
    compute_rag_vector_index(**config["parameters"])