import os
import json
import psutil
import asyncio
from loguru import logger

from litellm import acompletion
from tqdm.asyncio import tqdm
from pydantic import BaseModel

from second_brain_offline.utils import clip_tokens
from second_brain_offline.domain import Document
from second_brain_offline.config import settings

import litellm
litellm._turn_on_debug()

class SummarizerResponseFormat(BaseModel):
    summary: str

class SummarizationAgent:
    
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

    SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant specialized in summarizing documents.
Your task is to create a clear, concise TL;DR summary in markdown format.
Things to keep in mind while summarizing:
- titles of sections and sub-sections
- tags such as Generative AI, LLMs, etc.
- entities such as persons, organizations, processes, people, etc.
- the style such as the type, sentiment and writing style of the document
- the main findings and insights while preserving key information and main ideas
- ignore any irrelevant information such as cookie policies, privacy policies, HTTP errors,etc.

Document content:
{content}

Generate a concise TL;DR summary having a maximum of {characters} characters of the key findings from the provided documents, highlighting the most significant insights and implications.
Return the document in markdown format regardless of the original format.
"""
    
    def __init__(
        self,
        max_characters: int,
        model_id: str = "gpt-4o-mini",
        mock: bool = False,
        concurrent_requests: int = 10,
    ) -> None:
        self.max_characters = max_characters
        self.model_id = model_id
        self.mock = mock
        self.concurrent_requests = concurrent_requests
        
    def __call__(
        self,
        documents: Document | list[Document],
        temperature: float = 0.0,
    ) -> Document | list[Document]:
        
        """Process single document or batch of documents for summarization.

        Args:
            documents: Single Document or list of Documents to summarize.
            temperature: Temperature for the summarization model.
        Returns:
            Document | list[Document]: Processed document(s) with summaries.
        """
        
        is_single_document = isinstance(documents, Document)
        docs_list = [documents] if is_single_document else documents
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            results = asyncio.run(self.__get_summarized_batch(docs_list, temperature))
        else:
            results = loop.run_until_complete(self.__get_summarized_batch(docs_list, temperature))
            
        return results[0] if is_single_document else results
            
    async def __get_summarized_batch(
        self, 
        documents: list[Document],
        temperature: float = 0.0,
    ) -> list[Document]:
        
        """Asynchronously summarize multiple documents.

        Args:
            documents: List of documents to summarize.
            temperature: Temperature for the summarization model.
        Returns:
            list[Document]: Documents with generated summaries.
        """
        
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss
        total_documents = len(documents)
        logger.debug(
            f"Starting summarization batch with {self.concurrent_requests} concurrent requests. "
            f"Current process memory usage: {start_mem // (1024 * 1024)} MB"
        )
        
        summarized_documents = await self.__summarize_batch(documents, temperature, await_time_seconds = 7)
        
        logger.debug(f"Summarized documents raw: {len(summarized_documents)}")
        if summarized_documents is None:
            logger.error("__summarize_batch returned None!")
            return []
        if any(doc is None for doc in summarized_documents):
            logger.warning("Some docs in summarized_documents are None!")
        
        documents_with_summary = [
            doc for doc in summarized_documents if doc.summary is not None
        ]
        
        documents_without_summary = [
            doc for doc in summarized_documents if doc.summary is None
        ]
        
        if documents_without_summary:
            logger.info(
                f"Retrying {len(documents_without_summary)} failed documents with increased await time..."
            )
            retry_results = await self.__summarize_batch(
                documents_without_summary, temperature, await_time_seconds=20
            )
            
            retry_successes = [doc for doc in retry_results if doc.summary is not None]
            documents_with_summary.extend(retry_successes)
            final_failures = [doc for doc in retry_results if doc.summary is None]
            
            
        end_mem = process.memory_info().rss
        memory_diff = end_mem - start_mem
        logger.debug(
            f"Summarization batch completed. "
            f"Final process memory usage: {end_mem // (1024 * 1024)} MB, "
            f"Memory diff: {memory_diff // (1024 * 1024)} MB"
        )
                
        success_count = len(
            [doc for doc in documents_with_summary if doc.summary is not None]
        )
        
        failed_count = total_documents - success_count
        logger.info(
            f"Summarization completed: "
            f"{success_count}/{total_documents} succeeded ✓ | "
            f"{failed_count}/{total_documents} failed ✗"
        )

        return documents_with_summary
            
        
    async def __summarize_batch(
        self,
        documents: list[Document],
        temperature: float, 
        await_time_seconds: int,
    ) -> list[Document]:
        """Process a batch of documents with specified await time.

        Args:
            documents: List of documents to summarize.
            temperature: Temperature for the summarization model.
            await_time_seconds: Time to wait between requests.
        Returns:
            list[Document]: Processed documents with summaries.
        """
        
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        tasks = [self.__summarize_document(doc, semaphore, temperature, await_time_seconds)
            for doc in documents]

        results = []
    
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(documents),
            desc="summarizing documents"
        ):
            result = await coro
            results.append(result)
        
        return results
            
            
    async def __summarize_document(
        self,
        document: Document,
        semaphore: asyncio.Semaphore,
        temperature: float, 
        await_time_seconds: int = 5,
    ):
        """Generate a summary for a single document.

        Args:
            document: The Document object to summarize.
            semaphore: Optional semaphore for controlling concurrent requests.
            temperature: Temperature for the summarization model.
        Returns:
            Document | None: Document with generated summary or None if failed.
        """
        
        if self.mock:
            return document.add_summary("This is a mock summary")
        
        async def process_document():
            
            input_user_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
                characters=self.max_characters, content=document.content
            )
            
            # try:
            #     input_user_prompt = clip_tokens(  
            #         text=input_user_prompt,
            #         max_tokens=8192,
            #         model_id=self.model_id,
            #     )
            # except Exception as e:
            #     logger.warning(
            #         f"Failed to clip tokens for document {document.id}: {str(e)}"
            #     )
                
            messages = [{"content": input_user_prompt, "role": "user"}]

            
            try:
                response = await acompletion(
                    model=self.model_id,
                    messages=messages,
                    stream=False,
                    api_base="http://localhost:11434",
                    temperature=temperature
                )
                
                await asyncio.sleep(await_time_seconds)
                
                if not response.choices:
                    logger.warning(f"No summary generated for document: {document.id}")
                    return document
                
                summary = response.choices[0].message.content
                # summary = self.__parse_model_response(raw_answer)

                
                # if not summary:
                #     logger.warning(f"Failed to parse model response for document, {document.id}")
                #     return document
                                
                return document.add_summary(summary)
                
            except Exception as e:
                logger.error(f"Failed to generate summary for document: {document.id}.")
                return document
            
        
        if semaphore:
            async with semaphore:
                return await process_document()
        else:
            return await process_document()
                 
    # def __parse_model_response(self, 
    #     answer: str
    # ) ->str :
    #     if not answer:
    #         return None
        
    #     try:          
    #         parsed_summary = json.loads(answer)
    #         return SummarizerResponseFormat(
    #             parsed_summary["summary"]
    #         )
            
    #     except Exception:
    #         return None
