"""Async text splitter with contextual summarization support."""

import asyncio
from typing import Literal, Union, Callable

from loguru import logger
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from second_brain_offline.application.agents import (
    ContextualSummarizerAgent,
    SimpleSummarizerAgent,
)

SummarizationType = Literal["none", "simple", "contextual"]
SummarizerAgent = Union[ContextualSummarizerAgent, SimpleSummarizerAgent]


# NEW FUNCTION
async def get_splitter_async(
    chunk_size: int,
    summarization_type: SummarizationType = "none",
    **kwargs,
) -> "AsyncRecursiveCharacterTextSplitter":
    """Returns an async token-based text splitter.
    
    CHANGE: New async version of get_splitter.
    PROBLEM SOLVED: Returns async-compatible splitter.

    Args:
        chunk_size: Number of tokens for each text chunk.
        summarization_type: Type of summarization to use.
        **kwargs: Additional keyword arguments for summarization agent.

    Returns:
        AsyncRecursiveCharacterTextSplitter: Configured async text splitter.
    """
    
    chunk_overlap = int(0.15 * chunk_size)
    
    logger.info(
        f"Getting async splitter with chunk_size={chunk_size}, "
        f"overlap={chunk_overlap}, type={summarization_type}"
    )
    
    if summarization_type == "none":
        return AsyncRecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    if summarization_type == "simple":
        handler = SimpleSummarizerAgent(**kwargs)
    elif summarization_type == "contextual":
        handler = ContextualSummarizerAgent(**kwargs)
    else:
        raise ValueError(f"Invalid summarization_type: {summarization_type}")
    
    return AsyncRecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        handler=handler,
    )


# RENAMED AND MODIFIED CLASS
class AsyncRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """Async text splitter with optional contextual summarization.
    
    CHANGE: Renamed from HandlerRecursiveCharacterTextSplitter.
    CHANGE: Added async methods to prevent event loop blocking.
    PROBLEM SOLVED:
    - Old version blocked event loop when handler was async
    - New version properly handles async handlers
    """
    
    def __init__(
        self,
        handler: Callable | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the async splitter.

        Args:
            handler: Optional async/sync callable for post-processing chunks.
            *args: Additional positional arguments for parent class.
            **kwargs: Additional keyword arguments for parent class.
        """
        super().__init__(*args, **kwargs)
        
        self.handler = handler
        # NEW: Detect if handler is async
        self._is_async_handler = (
            asyncio.iscoroutinefunction(handler)
            if handler else False
        )
    
    def split_text(self, text: str) -> list[str]:
        """Synchronous split - raises error if handler is async.
        
        CHANGE: Added check to prevent misuse.
        PROBLEM SOLVED: Prevents blocking async handler in sync context.
        """
        # NEW: Prevent async handler in sync context
        if self._is_async_handler:
            raise RuntimeError(
                "Cannot use sync split_text() with async handler. "
                "Use split_text_async() instead."
            )
        
        chunks = super().split_text(text)
        
        if self.handler:
            return self.handler(text, chunks)
        
        return chunks
    
    async def split_text_async(self, text: str) -> list[str]:  # NEW METHOD
        """Async split with optional handler processing.
        
        CHANGE: New async version.
        PROBLEM SOLVED:
        - Old: self.handler(text, chunks) blocked if handler was async
        - New: await self.handler(text, chunks) properly awaits
        
        Args:
            text: Input text to split.

        Returns:
            list[str]: Processed text chunks.
        """
        # Base splitting is sync (fast tokenization) - run in thread pool
        chunks = await asyncio.to_thread(super().split_text, text)
        
        if not self.handler:
            return chunks
        
        # CHANGE: Properly handle async handlers
        if self._is_async_handler:
            return await self.handler(text, chunks)  # ASYNC
        else:
            return await asyncio.to_thread(self.handler, text, chunks)  # SYNC in thread
    
    async def split_documents_async(  # NEW METHOD
        self,
        documents: list[LangchainDocument],
    ) -> list[LangchainDocument]:
        """Split multiple documents asynchronously.
        
        CHANGE: New method for async document splitting.
        PROBLEM SOLVED:
        - Old: No async version, blocks event loop
        - New: Processes documents concurrently

        Args:
            documents: List of LangChain documents to split.

        Returns:
            list[LangchainDocument]: Chunked documents with preserved metadata.
        """
        all_chunks = []
        
        # CHANGE: Process documents concurrently
        tasks = [
            self._split_single_document(doc)
            for doc in documents
        ]
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle errors
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Document splitting failed: {result}")
                continue
            all_chunks.extend(result)
        
        logger.debug(
            f"Split {len(documents)} documents into {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    async def _split_single_document(  # NEW METHOD
        self,
        document: LangchainDocument,
    ) -> list[LangchainDocument]:
        """Split a single document into chunks.

        Args:
            document: LangChain document to split.

        Returns:
            list[LangchainDocument]: Chunked documents.
        """
        chunks = await self.split_text_async(document.page_content)
        
        return [
            LangchainDocument(
                page_content=chunk,
                metadata=document.metadata.copy(),
            )
            for chunk in chunks
        ]


# KEEP OLD FUNCTION FOR BACKWARDS COMPATIBILITY
def get_splitter(
    chunk_size: int,
    summarization_type: SummarizationType = "none",
    **kwargs,
) -> RecursiveCharacterTextSplitter:
    """Synchronous splitter (legacy support).
    
    CHANGE: Keep for backwards compatibility but discourage use.
    
    For new code, use get_splitter_async() instead.
    """
    logger.warning(
        "Using legacy sync splitter. Consider migrating to "
        "get_splitter_async() for better performance."
    )
    
    chunk_overlap = int(0.15 * chunk_size)
    
    if summarization_type == "none":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    raise NotImplementedError(
        "Sync contextual summarization is no longer supported. "
        "Use get_splitter_async() instead."
    )


# REMOVED: Old HandlerRecursiveCharacterTextSplitter class
# Replaced by: AsyncRecursiveCharacterTextSplitter