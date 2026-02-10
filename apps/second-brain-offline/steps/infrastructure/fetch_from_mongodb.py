from loguru import logger
from typing import Annotated

from zenml import step, get_step_context

from second_brain_offline.domain import Document
from second_brain_offline.infrastructure.mongo import MongoDBService

@step
def fetch_from_mongodb(
    collection_name: str,
    limit: int
    ) -> Annotated[list[Document], "fetched_documents"]:
    
    with MongoDBService(model=Document, collection_name=collection_name) as service:
        logger.info(f"Fetching documents from the '{collection_name}' mongodb collection.")
        documents = service.fetch_documents(limit, query={"content_quality_score": {"$gt": 0.8}})
                
    step_context = get_step_context()
    
    step_context.add_output_metadata(
        output_name="fetched_documents",
        metadata={
            "count" : len(documents),
        },
    )
    
    return list(documents)     
        
        
    
    
    