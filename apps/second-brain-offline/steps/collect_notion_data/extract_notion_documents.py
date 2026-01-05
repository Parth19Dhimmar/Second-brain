from loguru import logger
from typing_extensions import Annotated

from zenml import step, get_step_context

from second_brain_offline.domain import Document, DocumentMetadata
from second_brain_offline.infrastructure.notion import NotionDocumentClient

@step
def extract_notion_documents(
    documents_metadata : list[DocumentMetadata]
    ) -> Annotated[list[Document], "notion_documents"]:
    """Extract information from multiple notion documents.

    Args:
        documents_metadata (DocumentMetadata): list of notion documents metadata to extract conetent from.

    Returns:
        : List of Document Objects with extracted content.
    """
    
    client = NotionDocumentClient()
    documents = []
    
    for document_metadata in documents_metadata:
        documents.append(client.extract_document(document_metadata))
        
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="notion_documents",
            metadata={
                "documents_len": len(documents)
            }
        )        
        
        return documents
    
    