from typing import Literal, Union
from loguru import logger

from .embedding import get_embedding_model, EmbeddingsModel, EmbeddingModelType
from .splitter import get_splitter
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from langchain_mongodb.retrievers.parent_document import MongoDBAtlasParentDocumentRetriever

from langchain_mongodb import MongoDBAtlasVectorSearch
from second_brain_offline.config import settings

RetrieverType = Literal["parent", "contextual"]
RetrieverModel = Union[MongoDBAtlasParentDocumentRetriever, MongoDBAtlasHybridSearchRetriever]


def  get_retriever(
    embedding_model_id: str,
    embedding_model_type: EmbeddingModelType = "huggingface",
    retriever_type: RetrieverType = "contextual",
    k: int = 3,
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
) -> MongoDBAtlasHybridSearchRetriever:
    
   vector_store = MongoDBAtlasVectorSearch.from_connection_string(
       connection_string=settings.MONGODB_URI,
       embedding=embedding_model,
       namespace=f"{settings.MONGODB_DATABASE_NAME}.rag",
       text_key="chunk",
       embedding_key="embedding",
       relevance_score_fn="dotProduct"
   ) 
   
   retriever = MongoDBAtlasHybridSearchRetriever(
       vectorstore=vector_store,
       search_index_name="chunk_text_search",
       k=k,
       vector_penalty=50,
       fulltext_penalty=50,
   )
   
   return retriever