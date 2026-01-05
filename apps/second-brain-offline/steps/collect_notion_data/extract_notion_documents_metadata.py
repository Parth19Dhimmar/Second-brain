from loguru import logger
from typing import Annotated

from zenml import step, get_step_context

from second_brain_offline.domain import DocumentMetadata
from second_brain_offline.infrastructure.notion import NotionDatabaseClient


@step
def extract_notion_document_metadata(
    database_id : str,
) -> Annotated[list[DocumentMetadata], "notion_documents_metadata"]:
    """Extract the notion document metadta for specific notion databse
    
    Args:
        database_id : ID of notion datbase to query
        
    Returns:
        List of DocumentMetadata Object containing information about notion documents.
    
    """
    
    client = NotionDatabaseClient()
    documents_metadata = client.query_notion_database(database_id)
    
    logger.info(
        f"Extracted {len(DocumentMetadata)} documents metdata from {database_id}"
    )
    
    step_context = get_step_context()
    step_context.add_document_metadata(
        output_name="notion_documents_metadata",
        metadata={
            "database_id": database_id,
            "len_document_metadata": len(documents_metadata)
        } 
    )
    
    return documents_metadata