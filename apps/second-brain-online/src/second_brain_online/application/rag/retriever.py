import time 
from typing import Literal, Union
from loguru import logger

from .embedding import get_embedding_model, EmbeddingsModel, EmbeddingModelType
from .splitter import get_splitter
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from langchain_mongodb.retrievers.parent_document import MongoDBAtlasParentDocumentRetriever

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from langchain_mongodb import MongoDBAtlasVectorSearch
from second_brain_online.config import settings

RetrieverType = Literal["parent", "contextual"]
RetrieverModel = Union[MongoDBAtlasParentDocumentRetriever, MongoDBAtlasHybridSearchRetriever]


def get_retriever(
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType = "huggingface",
    retriever_type: RetrieverType = "contextual",
    k: int = 5,
    device: str = "cpu",
)-> RetrieverModel:
    """_summary_

    Args:
        embedding_model_id (str): _description_
        embedding_model_type (str): _description_
        retriever_type (str): _description_

    Returns:
        RetrieverModel: _description_
    """
    
    logger.info(
        f"Getting '{retriever_type}' retriever for '{embedding_model_type}' - '{embedding_model_id}' on '{device}' "
        f"with {k} top results"
    )
    
    embedding_model = get_embedding_model(
        embedding_model_id,
        embedding_model_type,
        device
    )
    
    
    if retriever_type == "parent":
        return get_parent_document_retriever(embedding_model, k)
    elif retriever_type == "contextual":
        return get_hybrid_search_retriever(embedding_model, k)
    else:
        raise ValueError(f"Invalid retriever type, {retriever_type}")
    

def get_parent_document_retriever(
    embedding_model: EmbeddingsModel,
    k: int,
) -> MongoDBAtlasParentDocumentRetriever:
    retriever = MongoDBAtlasParentDocumentRetriever.from_connection_string(
        connection_string=settings.MONGODB_URI,
        child_splitter=get_splitter(200),
        parent_splitter=get_splitter(800),
        embedding_model=embedding_model,
        database_name=settings.MONGODB_DATABASE_NAME,
        collection_name="rag",
        text_key="page_content",
        search_kwargs = { "k": k },
    )
    return retriever
    
def get_hybrid_search_retriever(
    embedding_model: EmbeddingsModel,
    k: int,
    reranker_top_n: int = 3,
) -> MongoDBAtlasHybridSearchRetriever:
    
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
       connection_string=settings.MONGODB_URI,
       embedding=embedding_model,
       namespace=f"{settings.MONGODB_DATABASE_NAME}.rag",
       text_key="chunk",
       embedding_key="embedding",
       relevance_score_fn="dotProduct"
    ) 

    base_retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store,
        search_index_name="chunk_text_search",
        k=k,          # fetch Top-N (e.g. 20) before reranking
        vector_penalty=50,
        fulltext_penalty=50,
    )

    # Reranker compresses Top-N → Top-K
    compressor = FlashrankRerank(top_n=reranker_top_n) # model_name="ms-marco-TinyBERT-L-2-v2"

    # reranking_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=base_retriever,
    # )
    # return reranking_retriever
    
    # ── Timed compression retriever ───────────────────────────────────────────
    # We wrap ContextualCompressionRetriever to intercept and time each phase:
    #   Phase 1: base_retriever.invoke()  → gte-large embedding + Atlas search
    #   Phase 2: compressor.compress()    → Flashrank reranking on CPU
    # This tells us exactly which phase is slow.
    class TimedContextualCompressionRetriever(ContextualCompressionRetriever):
        def invoke(self, query: str, **kwargs):
            # ── Phase 1: embedding + Atlas hybrid search ──────────────────────
            # gte-large embeds the query here on CPU — likely the slowest step.
            # Atlas then does vector + fulltext search and returns k=20 docs.
            t1_start = time.perf_counter()
            docs = self.base_retriever.invoke(query, **kwargs)
            t1_end = time.perf_counter()
            logger.info(
                f"[TIMING] Phase 1 — gte-large embed + Atlas hybrid search: "
                f"{t1_end - t1_start:.3f}s | {len(docs)} docs fetched"
            )
 
            # ── Phase 2: Flashrank reranking ──────────────────────────────────
            # Runs locally on CPU — reranks k=20 docs down to top_n=3.
            t2_start = time.perf_counter()
            compressed = self.base_compressor.compress_documents(docs, query)
            t2_end = time.perf_counter()
            logger.info(
                f"[TIMING] Phase 2 — Flashrank reranking: "
                f"{t2_end - t2_start:.3f}s | {len(compressed)} docs after rerank"
            )
 
            return compressed
 
    return TimedContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    