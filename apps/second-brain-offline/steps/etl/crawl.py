from typing_extensions import Annotated
from loguru import logger

from zenml import step, get_step_context

from second_brain_offline.domain import Document
from second_brain_offline.application.crawlers import Crawl4AICrawler

@step
def crawl(
    max_workers: int,
    documents: list[Document],
) -> Annotated[list[Document], "crawled_documents"]:
    """Crawl the child URLs of each document.

    Args:
        max_workers: Maximum number of concurrent requests. Defaults to 10.
        documents: List of documents to crawl and extract child URLs from.

    Returns:
        list[Document]: List containing original documents plus newly crawled child documents.
    """
    
    crawler = Crawl4AICrawler(max_concurrent_requests=max_workers)
    
    child_pages = crawler(documents)
    
    # crawled will have both notion documents + crwaled links documents
    extended_documents = documents.copy()
    extended_documents.extend(child_pages)
    extended_documents = list(set(extended_documents))
    
    logger.info(f"Before crawling, documents count : {len(documents)}")
    logger.info(f"After Crawling, documents count: {len(extended_documents)}")
    logger.info(f"New documents added count: {len(extended_documents) - len(documents)}")
    
    step_context = get_step_context()
    
    step_context.add_output_metadata(
        output_name="crawled_documents",
        metadata={      
            "len_documents_before_crawling": len(documents),
            "len_documents_after_crawling": len(extended_documents),
            "len_documents_new": len(extended_documents) - len(documents),
        }
    )
    
    return extended_documents
    