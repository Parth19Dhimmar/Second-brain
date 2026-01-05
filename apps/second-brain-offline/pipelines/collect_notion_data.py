from pathlib import Path

from loguru import logger
from zenml import pipeline

from steps.collect_notion_data import (
    extract_notion_document_metadata,
    extract_notion_documents
)
from steps.infrastructure import save_documents_to_disk, upload_to_s3

@pipeline
def collect_notion_data(
    database_ids : list[str], data_dir: Path, to_s3: bool = False
) -> None :
    notion_data_dir = data_dir / "notion"
    notion_data_dir.mkdir(parents=True, exist_ok=True)
    
    invocation_ids = []
    for index, database_id in enumerate(database_ids):
        logger.info(f"Collecting pages from database '{database_id}'")
        
        documents_metadata = extract_notion_document_metadata(database_id=database_id)
        document_content = extract_notion_documents(documents_metadata=documents_metadata)
        
        result = save_documents_to_disk(
                documents=document_content,
                data_dir=notion_data_dir,
            )
        invocation_ids.append(result.invocation_id)
    
    if to_s3:
        upload_to_s3(
            folder_path=notion_data_dir,
            s3_prefix="second_brain/notion",
            after=invocation_ids,
        )