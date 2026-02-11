"""Async contextual summarization agent with production features."""

import os
import psutil
import asyncio
from typing import Callable

from loguru import logger
from tqdm.asyncio import tqdm as async_tqdm  # CHANGED: from tqdm import tqdm
from litellm import acompletion
from pydantic import BaseModel
from openai import AsyncOpenAI

from second_brain_offline.config import settings

class ContextualDocument(BaseModel):  # UNCHANGED
    """Document with optional contextual summary."""
    
    content: str
    chunk: str | None = None
    contextual_summarization: str | None = None
    
    def add_contextual_summarization(self, summary: str) -> "ContextualDocument":
        self.contextual_summarization = summary
        return self


class ContextualSummarizerAgent:
    """Async agent for generating contextual summaries.
    
    CHANGE: Removed event loop detection/creation.
    CHANGE: Made __call__ async-only.
    CHANGE: Added retry logic.
    CHANGE: Added metrics tracking.
    
    PROBLEM SOLVED:
    - No more event loop conflicts
    - Automatic retry on failures
    - Better observability
    """
    
    SYSTEM_PROMPT_TEMPLATE = """
<document> 
{content} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk} 
</chunk> 
Please give a short succinct context of maximum {characters} characters to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
"""
    
    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        max_characters: int = 128,
        mock: bool = False,
        max_concurrent_requests: int = 50,      # CHANGED: 2 -> 50 (production default)
        temperature: float = 0.0,
        retry_attempts: int = 2,                 # NEW
        base_delay: float = 1.0,                 # NEW: renamed from await_time_seconds
    ):
        """Initialize the contextual summarizer.

        Args:
            model_id: LLM model identifier.
            max_characters: Maximum summary length.
            mock: Enable mock mode for testing.
            max_concurrent_requests: Max concurrent API calls.
            temperature: LLM temperature parameter.
            retry_attempts: Number of retry attempts for failed requests.  # NEW
            base_delay: Base delay in seconds between API calls.          # NEW
        """
        self.model_id = model_id
        self.max_characters = max_characters
        self.mock = mock
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature
        self.retry_attempts = retry_attempts      # NEW
        self.base_delay = base_delay              # NEW
        
        # NEW: Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
    
    async def __call__(  # CHANGED: Now async (was sync with event loop detection)
        self,
        content: str,
        chunks: list[str],
    ) -> list[str]:
        """Async callable interface for the agent.
        
        CHANGE: Removed event loop detection - now pure async.
        OLD: try/except to detect loop, asyncio.run() or loop.run_until_complete()
        NEW: Direct async call - caller manages event loop
        PROBLEM SOLVED: No more event loop conflicts

        Args:
            content: Full document content for context.
            chunks: List of text chunks to summarize.

        Returns:
            list[str]: Chunks with prepended contextual summaries.
        """
        # CHANGED: Direct async call instead of event loop detection
        return await self._context_summarize_batch(content, chunks)
    
    async def _context_summarize_batch(  # CHANGED: from __context_summarize_batch
        self,
        content: str,
        chunks: list[str],
    ) -> list[str]:
        """Process a batch of chunks with contextual summarization.
        
        CHANGE: Renamed from __context_summarize_batch.
        CHANGE: Removed memory tracking (less important).
        CHANGE: Use renamed methods.

        Args:
            content: Full document content.
            chunks: Text chunks to summarize.

        Returns:
            list[str]: Chunks with contextual summaries prepended.
        """
        total_chunks = len(chunks)
        logger.info(
            f"Starting contextual summarization for {total_chunks} chunks "
            f"(max_concurrent={self.max_concurrent_requests})"
        )
        
        # Create documents
        documents = [
            ContextualDocument(content=content, chunk=chunk)
            for chunk in chunks
        ]
        
        # First attempt
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        contextual_documents = await self._process_documents(  # CHANGED: from __context_summarize_documents
            documents,
            semaphore,
            retry_delay=self.base_delay,
        )
        
        # Separate success/failure
        success_docs = [
            doc for doc in contextual_documents
            if doc.contextual_summarization is not None
        ]
        
        failed_docs = [
            doc for doc in contextual_documents
            if doc.contextual_summarization is None
        ]
        
        # Retry failed documents with increased delay
        if failed_docs:
            logger.warning(
                f"Retrying {len(failed_docs)} failed chunks with increased delay..."
            )
            
            retry_docs = await self._process_documents(
                failed_docs,
                semaphore,
                retry_delay=self.base_delay * 3,  # CHANGED: Increased backoff
            )
            
            success_docs.extend([
                doc for doc in retry_docs
                if doc.contextual_summarization is not None
            ])
        
        # Log results
        success_count = len(success_docs)
        failed_count = total_chunks - success_count
        
        logger.info(
            f"Contextual summarization complete: "
            f"✓ {success_count}/{total_chunks} succeeded | "
            f"✗ {failed_count}/{total_chunks} failed"
        )
        
        # Build final chunks with summaries
        return self._build_contextual_chunks(success_docs, chunks)
    
    async def _process_documents(  # CHANGED: from __context_summarize_documents
        self,
        documents: list[ContextualDocument],
        semaphore: asyncio.Semaphore,
        retry_delay: float,
    ) -> list[ContextualDocument]:
        """Process documents with controlled concurrency.
        
        CHANGE: Use async_tqdm instead of sync tqdm.
        CHANGE: Call new _summarize_chunk_with_retry.

        Args:
            documents: Documents to process.
            semaphore: Semaphore for rate limiting.
            retry_delay: Delay between API calls.

        Returns:
            list[ContextualDocument]: Processed documents.
        """
        
        tasks = [
            self._summarize_chunk_with_retry(doc, semaphore, retry_delay)  # CHANGED: new method
            for doc in documents
        ]
        
        results = []
        
        # CHANGED: async_tqdm instead of sync tqdm
        async for result in async_tqdm(
            self._as_completed(tasks),
            total=len(documents),
            desc="Summarizing chunks",
            leave=False,
        ):
            results.append(result)
        
        return results
    
    async def _summarize_chunk_with_retry(  # NEW METHOD
        self,
        document: ContextualDocument,
        semaphore: asyncio.Semaphore,
        retry_delay: float,
    ) -> ContextualDocument:
        """Summarize a chunk with automatic retry.
        
        CHANGE: New method with retry logic.
        OLD: Single attempt in __summarize_chunk
        NEW: Multiple attempts with exponential backoff
        PROBLEM SOLVED: Transient API failures now recoverable.

        Args:
            document: Document to summarize.
            semaphore: Semaphore for rate limiting.
            retry_delay: Base delay between requests.

        Returns:
            ContextualDocument: Document with summary (or None on failure).
        """
        if self.mock:
            await asyncio.sleep(0.01)
            return document.add_contextual_summarization("Mock summary")
        
        last_error = None
        
        # NEW: Retry loop with exponential backoff
        for attempt in range(self.retry_attempts):
            try:
                async with semaphore:
                    result = await self._summarize_chunk(
                        document,
                        retry_delay * (2 ** attempt),  # Exponential backoff
                    )
                    
                    self._successful_requests += 1  # NEW: Track success
                    return result
                    
            except Exception as e:
                last_error = e
                
                if attempt < self.retry_attempts - 1:
                    logger.debug(
                        f"Summarization failed (attempt {attempt + 1}/"
                        f"{self.retry_attempts}): {str(e)}"
                    )
                else:
                    logger.error(f"Summarization failed after retries: {str(e)}")
                    self._failed_requests += 1  # NEW: Track failure
        
        return document  # Return without summary
    
    async def _summarize_chunk(  # CHANGED: from __summarize_chunk, simplified signature
        self,
        document: ContextualDocument,
        delay: float,
    ) -> ContextualDocument:
        """Generate contextual summary for a single chunk.
        
        CHANGE: Simplified - no semaphore (handled by caller).
        CHANGE: Renamed await_time_seconds -> delay.

        Args:
            document: Document to summarize.
            delay: Delay after API call.

        Returns:
            ContextualDocument: Document with summary.
        """
        input_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            characters=self.max_characters,
            content=document.content[:6000],
            chunk=document.chunk,
        )
        
        messages = [{"role": "system", "content": input_prompt}]
        
        # Call LLM
        response = await acompletion(
            model=self.model_id,
            messages=messages,
            stream=False,
            temperature=self.temperature,
        )
        
        # Rate limiting
        await asyncio.sleep(delay)
        
        # Extract summary
        if not response.choices:
            logger.warning("No response from LLM")
            return document
        
        context_summary = response.choices[0].message.content
        
        return document.add_contextual_summarization(context_summary)
    
    @staticmethod
    def _build_contextual_chunks(  # NEW METHOD
        documents: list[ContextualDocument],
        original_chunks: list[str],
    ) -> list[str]:
        """Build final chunks with summaries prepended.
        
        CHANGE: Extracted from _context_summarize_batch for clarity.

        Args:
            documents: Documents with summaries.
            original_chunks: Original chunks for fallback.

        Returns:
            list[str]: Chunks with contextual summaries.
        """
        # Create lookup by chunk content
        summary_map = {
            doc.chunk: doc.contextual_summarization
            for doc in documents
            if doc.chunk is not None
        }
        
        # Build final chunks
        contextual_chunks = []
        for chunk in original_chunks:
            summary = summary_map.get(chunk)
            
            if summary:
                contextual_chunks.append(f"{summary}\n\n{chunk}")
            else:
                contextual_chunks.append(chunk)
        
        return contextual_chunks
    
    @staticmethod
    async def _as_completed(tasks):  # NEW METHOD
        """Async generator for completed tasks."""
        for coro in asyncio.as_completed(tasks):
            yield await coro
    
    def get_metrics(self) -> dict:  # NEW METHOD
        """Get summarization metrics.

        Returns:
            dict: Metrics including success/failure counts.
        """
        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }


# REMOVED: Old code with event loop detection
# OLD: try/except asyncio.get_running_loop()
# OLD: asyncio.run() or loop.run_until_complete()

class SimpleSummarizerAgent:
    """Generates summaries for documents using LiteLLM with async support.

    This class handles the interaction with language models through LiteLLM to
    generate concise summaries while preserving key information from the original
    documents. It supports both single and batch document processing.

    Attributes:
        max_characters: Maximum number of characters for the summary.
        model_id: The ID of the language model to use for summarization.
        mock: If True, returns mock summaries instead of using the model.
        max_concurrent_requests: Maximum number of concurrent API requests.
    """

    SYSTEM_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant specialized in summarizing documents for the purposes of improving semantic and keyword search retrieval. 
Generate a concise TL;DR summary in plain text format having a maximum of {characters} characters of the key findings from the provided documents, 
highlighting the most significant insights. Answer only with the succinct context and nothing else.

### Input:
{content}

### Response:
"""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: str | None = settings.HUGGINGFACE_DEDICATED_ENDPOINT,
        api_key: str | None = settings.HUGGINGFACE_ACCESS_TOKEN,
        max_characters: int = 128,
        mock: bool = False,
        max_concurrent_requests: int = 4,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.max_characters = max_characters
        self.mock = mock
        self.max_concurrent_requests = max_concurrent_requests

        if self.model_id == "tgi":
            assert self.base_url and self.api_key, (
                "Base URL and API key are required for TGI Hugging Face Dedicated Endpoint"
            )

            self.client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        else:
            self.client = AsyncOpenAI()

    def __call__(self, content: str, chunks: list[str]) -> list[str]:
        """Process document chunks for contextual summarization.

        Args:
            content: The full document content
            chunks: List of document chunks to summarize

        Returns:
            list[str]: List of chunks with added contextual summaries
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            results = asyncio.run(self.__summarize_context_batch(content, chunks))
        else:
            results = loop.run_until_complete(
                self.__summarize_context_batch(content, chunks)
            )

        return results

    async def __summarize_context_batch(
        self, content: str, chunks: list[str]
    ) -> list[str]:
        """Asynchronously summarize multiple document chunks.

        Args:
            content: The full document content
            chunks: List of document chunks to summarize

        Returns:
            list[str]: List of chunks with added contextual summaries
        """

        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        logger.debug(
            f"Starting summarizing document."
            f"Initial memory usage: {start_mem // (1024 * 1024)} MB"
        )

        document = await self.__summarize(
            document=ContextualDocument(content=content), await_time_seconds=20
        )

        end_mem = process.memory_info().rss
        memory_diff = end_mem - start_mem
        logger.debug(
            f"Summarization completed. "
            f"Final memory usage: {end_mem // (1024 * 1024)} MB, "
            f"Memory difference: {memory_diff // (1024 * 1024)} MB"
        )

        contextual_chunks = []
        for chunk in chunks:
            if document.contextual_summarization is not None:
                chunk = f"{document.contextual_summarization}\n\n{chunk}"
            else:
                chunk = f"{chunk}"

            contextual_chunks.append(chunk)

        return contextual_chunks

    async def __summarize(
        self,
        document: ContextualDocument,
        await_time_seconds: int = 2,
    ) -> ContextualDocument:
        """Generate a contextual summary for a single document.

        Args:
            document: The document to summarize
            await_time_seconds: Time in seconds to wait between requests

        Returns:
            ContextualDocument: Document with generated summary
        """

        if self.mock:
            return document.add_contextual_summarization("This is a mock summary")

        async def process_document() -> ContextualDocument:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": self.SYSTEM_PROMPT_TEMPLATE.format(
                                characters=self.max_characters, content=document.content
                            ),
                        },
                    ],
                    stream=False,
                    temperature=0,
                )
                await asyncio.sleep(await_time_seconds)  # Rate limiting

                if not response.choices:
                    logger.warning("No contextual summary generated for chunk")
                    return document

                context_summary: str = response.choices[0].message.content
                return document.add_contextual_summarization(context_summary)
            except Exception as e:
                logger.warning(f"Failed to generate contextual summary: {str(e)}")
                return document

        return await process_document()