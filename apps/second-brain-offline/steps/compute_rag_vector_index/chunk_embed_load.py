
from typing import Any, Generator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from zenml import step, get_step_context
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb.retrievers import MongoDBAtlasParentDocumentRetriever

from second_brain_offline.domain import Document
from second_brain_offline.application.rag import (
    get_splitter,
    get_retriever,
    EmbeddingModelType,
    SummarizationType,
    RetrieverType
)
from second_brain_offline.infrastructure.mongo import MongoDBService, MongoDBIndex

@step
def chunk_embed_load(
    documents: list[Document],
    embed_collection_name: str,
    retriever_type: RetrieverType,
    contextual_summarization_type: SummarizationType,
    contextual_agent_model_id: str,   
    contextual_agent_max_characters: int,
    embedding_model_type: EmbeddingModelType,
    embedding_model_id: str,
    embedding_model_dim: int,
    processing_max_workers: int,
    processing_batch_size: int,
    mock: bool, 
    chunk_size: int, 
    device: str = "cpu",
):
    splitter = get_splitter(
        chunk_size=chunk_size,
        summarization_type=contextual_summarization_type,
        model_id=contextual_agent_model_id,
        max_characters=contextual_agent_max_characters,
        mock=mock,
        max_concurrent_requests=processing_max_workers
    )
    
    logger.info(f"Splitter : , {splitter}")
    
    logger.info(f"""
            embedding_model_id : {embedding_model_id},
            embedding_model_type : {embedding_model_type},
            retriever_type : {retriever_type},
        """)
    
    retriever = get_retriever(
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
        device=device
    )
    
    logger.info(f"retriever : {retriever}")
    
    langchain_documents = [
        LangchainDocument(
            page_content=document.content, 
            metadata=document.metadata.model_dump(),
        )
        for document in documents
        if document
    ]
    
    # call process_docs
    with MongoDBService(
        model=Document,
        collection_name=embed_collection_name,
    ) as mongodb_client:
        mongodb_client.clear_collection()
    
        process_docs(
            retriever,
            splitter,
            documents=langchain_documents,
            batch_size=processing_batch_size,
            max_workers=processing_max_workers
        )
        
        index = MongoDBIndex(
            retriever=retriever,
            mongodb_client=mongodb_client
        )
        
        # create mongo index
        index.create(
            embedding_dim=embedding_model_dim,
            is_hybrid=retriever_type == "contextual"
        )
    

def process_docs(
    retriever: Any,
    splitter: RecursiveCharacterTextSplitter,
    documents: list[LangchainDocument],
    batch_size: int = 4,
    max_workers: int = 2,
) -> list[None]:
    """Process LangChain documents into MongoDB using thread pool.

    Args:
        retriever: MongoDB Atlas document retriever instance.
        splitter: Text splitter instance for chunking documents.
        documents: List of LangChain documents to process.
        batch_size: Number of documents to process in each batch.
        max_workers: Maximum number of concurrent threads.

    Returns:
        List of None values representing completed batch processing results.
    """
    doc_batches = list(get_batches(
        documents, batch_size
    ))
    
    results = []
    total_docs = len(documents)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_batch, retriever, splitter, batch)
            for batch in doc_batches
        ]
        with tqdm(total=total_docs, desc="Processing docs") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(batch_size)
    return results
            
                
def process_batch(
    retriever: Any, 
    splitter: RecursiveCharacterTextSplitter,
    doc_batch: list[LangchainDocument],
):
    """Ingest batches of documents into MongoDB by splitting and embedding.

    Args:
        retriever: MongoDB Atlas document retriever instance.
        splitter: Text splitter instance for chunking documents.
        doc_batch: List of documents to ingest in this batch.

    Raises:
        Exception: If there is an error processing the batch of documents.
    """
    try:
        if isinstance(retriever, MongoDBAtlasParentDocumentRetriever):
            retriever.add_documents(doc_batch)
        else:
            split_docs = splitter.split_documents(doc_batch)
            retriever.vectorstore.add_documents(split_docs)
            
        logger.info(f"Successfully processed {len(doc_batch)} documents.")
    except Exception as e:
        logger.warning(f"Error processing batch of {len(doc_batch)} documents: {str(e)}")
        
        
def get_batches(
    documents: list[LangchainDocument],
    batch_size: int
) -> Generator[list[LangchainDocument], None, None]:
    """Return batches of documents to ingest into MongoDB.

    Args:
        documents: List of LangChain documents to batch.
        batch_size: Number of documents in each batch.

    Yields:
        Generator[list[LangChainDocument]]: Batches of documents of size batch_size.
    """
    for i in range(0, len(documents), batch_size):
        yield documents[i : i + batch_size]
            
    