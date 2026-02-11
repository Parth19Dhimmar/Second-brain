"""Async document processing step with production-grade error handling."""

import asyncio
from typing import Any
from contextlib import asynccontextmanager

from loguru import logger
from zenml import step, get_step_context
from langchain_core.documents import Document as LangchainDocument
from tqdm.asyncio import tqdm as async_tqdm

from second_brain_offline.domain import Document
from second_brain_offline.application.rag import (
    get_splitter_async,  # NEW: async version
    get_retriever,
    EmbeddingModelType,
    SummarizationType,
    RetrieverType,
)
from second_brain_offline.infrastructure.mongo import MongoDBService, MongoDBIndex


@step
def chunk_embed_load_async(  # RENAMED from chunk_embed_load
    documents: list[Document],
    embed_collection_name: str,
    retriever_type: RetrieverType,
    contextual_summarization_type: SummarizationType,
    contextual_agent_model_id: str | None,
    contextual_agent_max_characters: int | None,
    embedding_model_type: EmbeddingModelType,
    embedding_model_id: str,
    embedding_model_dim: int,
    processing_max_concurrent: int,  # RENAMED from processing_max_workers
    processing_batch_size: int,
    mock: bool,
    chunk_size: int,
    device: str = "cpu",
    enable_retry: bool = True,  # NEW
    max_retries: int = 3,       # NEW
):
    """Async step for chunking, embedding, and loading documents.
    
    CHANGE: Converted to async wrapper (ZenML steps must be sync).
    PROBLEM SOLVED: Single event loop, no thread conflicts.
    """
    
    # CHANGE: Run async pipeline (single event loop)
    asyncio.run(
        _async_pipeline(
            documents=documents,
            embed_collection_name=embed_collection_name,
            retriever_type=retriever_type,
            contextual_summarization_type=contextual_summarization_type,
            contextual_agent_model_id=contextual_agent_model_id,
            contextual_agent_max_characters=contextual_agent_max_characters,
            embedding_model_type=embedding_model_type,
            embedding_model_id=embedding_model_id,
            embedding_model_dim=embedding_model_dim,
            processing_max_concurrent=processing_max_concurrent,
            processing_batch_size=processing_batch_size,
            mock=mock,
            chunk_size=chunk_size,
            device=device,
            enable_retry=enable_retry,
            max_retries=max_retries,
        )
    )


async def _async_pipeline(  # NEW FUNCTION
    documents: list[Document],
    embed_collection_name: str,
    retriever_type: RetrieverType,
    contextual_summarization_type: SummarizationType,
    contextual_agent_model_id: str | None,
    contextual_agent_max_characters: int | None,
    embedding_model_type: EmbeddingModelType,
    embedding_model_id: str,
    embedding_model_dim: int,
    processing_max_concurrent: int,
    processing_batch_size: int,
    mock: bool,
    chunk_size: int,
    device: str,
    enable_retry: bool,
    max_retries: int,
):
    """Core async pipeline logic.
    
    CHANGE: New function containing all async logic.
    PROBLEM SOLVED: All async operations in single event loop.
    """
    
    logger.info(f"Starting async pipeline for {len(documents)} documents")
    
    # CHANGE: Use async splitter
    splitter = await get_splitter_async(
        chunk_size=chunk_size,
        summarization_type=contextual_summarization_type,
        model_id=contextual_agent_model_id,
        max_characters=contextual_agent_max_characters,
        mock=mock,
        max_concurrent_requests=processing_max_concurrent,
    )
    
    # CHANGE: Run sync retriever initialization in thread pool
    retriever = await asyncio.to_thread(
        get_retriever,
        embedding_model_id=embedding_model_id,
        embedding_model_type=embedding_model_type,
        retriever_type=retriever_type,
        device=device,
    )
    
    # Convert to LangChain documents (unchanged)
    langchain_documents = [
        LangchainDocument(
            page_content=doc.content,
            metadata=doc.metadata.model_dump(),
        )
        for doc in documents
        if doc and doc.content
    ]
    
    # CHANGE: Use async context manager
    async with _mongodb_context(embed_collection_name) as mongodb_client:
        await asyncio.to_thread(mongodb_client.clear_collection)
        
        # CHANGE: Use new async processor
        processor = AsyncDocumentProcessor(
            retriever=retriever,
            splitter=splitter,
            mongodb_client=mongodb_client,
            batch_size=processing_batch_size,
            max_concurrent=processing_max_concurrent,
            enable_retry=enable_retry,
            max_retries=max_retries,
        )
        
        await processor.process_documents(langchain_documents)
        
        # Create index
        index = MongoDBIndex(
            retriever=retriever,
            mongodb_client=mongodb_client,
        )
        
        await asyncio.to_thread(
            index.create,
            embedding_dim=embedding_model_dim,
            is_hybrid=(retriever_type == "contextual"),
        )
        
        logger.info("✅ Vector index created successfully")


@asynccontextmanager
async def _mongodb_context(collection_name: str):  # NEW FUNCTION
    """Async context manager for MongoDB.
    
    CHANGE: New async context manager.
    PROBLEM SOLVED: Proper async resource management.
    """
    mongodb_client = None
    try:
        mongodb_client = await asyncio.to_thread(
            lambda: MongoDBService(
                model=Document,
                collection_name=collection_name,
            ).__enter__()
        )
        yield mongodb_client
    finally:
        if mongodb_client:
            await asyncio.to_thread(
                mongodb_client.__exit__,
                None, None, None,
            )


class AsyncDocumentProcessor:  # NEW CLASS - replaces process_docs function
    """Async document processor with retry logic.
    
    CHANGE: Replaced ThreadPoolExecutor-based processing.
    PROBLEM SOLVED:
    - No more event loop conflicts
    - Proper async concurrency with semaphore
    - Automatic retry with exponential backoff
    - Better progress tracking
    """
    
    def __init__(
        self,
        retriever: Any,
        splitter: Any,
        mongodb_client: Any,
        batch_size: int,
        max_concurrent: int,
        enable_retry: bool = True,
        max_retries: int = 3,
    ):
        self.retriever = retriever
        self.splitter = splitter
        self.mongodb_client = mongodb_client
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)  # CHANGE: Semaphore instead of ThreadPool
        
        # Metrics
        self.successful_docs = 0
        self.failed_docs = 0
        self.total_chunks = 0
    
    async def process_documents(
        self,
        documents: list[LangchainDocument],
    ) -> None:
        """Process all documents with controlled concurrency.
        
        CHANGE: Async processing with semaphore.
        OLD: ThreadPoolExecutor with futures
        NEW: asyncio.gather with semaphore
        """
        
        logger.info(
            f"Processing {len(documents)} documents with "
            f"batch_size={self.batch_size}, max_concurrent={self.max_concurrent}"
        )
        
        batches = list(self._create_batches(documents, self.batch_size))
        
        # CHANGE: Async tasks instead of thread pool
        tasks = [
            self._process_batch_with_semaphore(batch, idx)
            for idx, batch in enumerate(batches)
        ]
        
        # CHANGE: async_tqdm instead of sync tqdm
        results = []
        async for result in async_tqdm(
            self._as_completed(tasks),
            total=len(tasks),
            desc="Processing batches",
            unit="batch",
        ):
            results.append(result)
        
        logger.info(
            f"Complete: ✓ {self.successful_docs} docs | "
            f"✗ {self.failed_docs} failed | "
            f"📊 {self.total_chunks} chunks"
        )
    
    async def _process_batch_with_semaphore(  # NEW METHOD
        self,
        batch: list[LangchainDocument],
        batch_idx: int,
    ) -> dict:
        """Process batch with semaphore control.
        
        CHANGE: Semaphore instead of thread pool.
        PROBLEM SOLVED: Proper async concurrency limiting.
        """
        async with self.semaphore:
            return await self._process_batch_with_retry(batch, batch_idx)
    
    async def _process_batch_with_retry(  # NEW METHOD
        self,
        batch: list[LangchainDocument],
        batch_idx: int,
    ) -> dict:
        """Process batch with exponential backoff retry.
        
        CHANGE: Added automatic retry logic.
        OLD: Single attempt, log error on failure
        NEW: Multiple attempts with exponential backoff
        PROBLEM SOLVED: Transient failures now recoverable.
        """
        
        last_error = None
        
        for attempt in range(self.max_retries if self.enable_retry else 1):
            try:
                result = await self._process_batch(batch, batch_idx)
                self.successful_docs += len(batch)
                return result
                
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"Batch {batch_idx} failed (attempt {attempt + 1}/"
                        f"{self.max_retries}): {str(e)}. Retrying in {backoff}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Batch {batch_idx} failed after {self.max_retries} attempts")
                    self.failed_docs += len(batch)
        
        return {"error": str(last_error), "batch_idx": batch_idx}
    
    async def _process_batch(  # MODIFIED METHOD
        self,
        batch: list[LangchainDocument],
        batch_idx: int,
    ) -> dict:
        """Process a single batch.
        
        CHANGE: Made async, uses async splitter.
        OLD: split_docs = splitter.split_documents(batch)  # BLOCKS event loop
        NEW: split_docs = await self.splitter.split_documents_async(batch)  # ASYNC
        PROBLEM SOLVED: No more blocking operations in event loop.
        """
        
        # CHANGE: Async splitting
        split_docs = await self.splitter.split_documents_async(batch)
        self.total_chunks += len(split_docs)
        
        # CHANGE: Check if vectorstore supports async
        if hasattr(self.retriever.vectorstore, 'aadd_documents'):
            await self.retriever.vectorstore.aadd_documents(split_docs)
        else:
            # CHANGE: Wrap blocking operation in thread pool
            await asyncio.to_thread(
                self.retriever.vectorstore.add_documents,
                split_docs,
            )
        
        return {
            "batch_idx": batch_idx,
            "docs_processed": len(batch),
            "chunks_created": len(split_docs),
        }
    
    @staticmethod
    def _create_batches(  # UNCHANGED
        documents: list[LangchainDocument],
        batch_size: int,
    ):
        """Create batches from documents."""
        for i in range(0, len(documents), batch_size):
            yield documents[i : i + batch_size]
    
    @staticmethod
    async def _as_completed(tasks):  # NEW METHOD
        """Async generator for completed tasks."""
        for coro in asyncio.as_completed(tasks):
            yield await coro


# REMOVED: Old process_docs, process_batch, get_batches functions
# They are replaced by AsyncDocumentProcessor class