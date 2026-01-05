from pathlib import Path
from zenml import pipeline
from loguru import logger

from steps.generate_dataset import (
    create_histogram, generate_summary_dataset
)
from steps.infrastructure import (
    fetch_from_mongodb, 
    upload_to_huggingface,
    save_dataset_to_disk,
)
from second_brain_offline.domain import Document, InstructDataset

@pipeline
def generate_dataset(
    mongo_collection_name: str,
    fetch_limit: int = 1000,
    load_dataset_id= str,
    summarization_max_characters: int = 256,
    summarization_model: str = "gpt-4o-mini",
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    max_workers: int = 10,
    mock: bool = False,
    min_document_length: int = 50,
    min_quality_score: float = 0.2,
    augmentation_loops: int = 4,
    output_dir: Path = Path("data/"),
) -> None:
    
    documents = fetch_from_mongodb(
        mongo_collection_name,
        fetch_limit,
    )
    
    # create_histogram(documents)
    
    summary_dataset = generate_summary_dataset(
        documents=documents,
        summarization_model=summarization_model,
        summarization_max_characters=summarization_max_characters,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        max_workers=max_workers,
        mock=mock,
        min_document_length=min_document_length,
        min_quality_score=min_quality_score,
        augmentation_loops=augmentation_loops,
        )
    
    upload_to_huggingface(dataset=summary_dataset, dataset_id=load_dataset_id)
    
    save_dataset_to_disk(
        dataset=summary_dataset,
        output_dir=output_dir / "dataset" / load_dataset_id
    )
    

if __name__ == "__main__":
    import yaml
    
    with open("configs/generate_dataset.yaml", "r") as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded configurations to run etl_precomputed pipeline: {config}")
    
    generate_dataset(
    **config["parameters"] 
) 
    
    
    
    
    
    
    
    
    
    
    

