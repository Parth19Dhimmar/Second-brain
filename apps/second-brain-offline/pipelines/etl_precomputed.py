from pathlib import Path

from loguru import logger
from zenml import pipeline

from steps.infrastructure import read_documents_from_disk, ingest_to_mongodb

@pipeline
def etl_precomputed(
    data_dir: Path,
    load_collection_name: str
    ) -> None:
    crawled_data_dir = data_dir / "crawled"
    documents = read_documents_from_disk(
        data_directory=crawled_data_dir,
        nesting_level=0
    )
    
    if documents:
        ingest_to_mongodb(
            models=documents,
            collection_name=load_collection_name
        )


if __name__ == "__main__":
    import yaml
    
    with open("configs/etl_precomputed.yaml", "r") as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded configurations to run etl_precomputed pipeline: {config}")
    
    etl_precomputed(
    **config["parameters"] 
)    