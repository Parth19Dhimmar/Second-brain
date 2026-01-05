from typing import Annotated

from zenml import step, get_step_context

from second_brain_offline.domain import Document, InstructDataset
from second_brain_offline.application.dataset import SummarizationDatasetGenerator

@step
def generate_summary_dataset(
    documents: list[Document],
    summarization_model: str,
    summarization_max_characters: int,
    val_split_ratio: float = 0.1,
    test_split_ratio: float = 0.1,
    max_workers: int = 10,
    mock: bool = False,
    min_document_length: int = 50,
    min_quality_score: float = 0.2,
    max_summary_length_factor: float = 2,
    augmentation_loops: int = 4,
) -> Annotated[InstructDataset, "summary_dataset"]:
    
    dataset_generator = SummarizationDatasetGenerator(
        summarization_model=summarization_model,
        summarization_max_characters=summarization_max_characters,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        max_workers=max_workers,
        mock=mock,
        min_document_length=min_document_length,
        min_quality_score=min_quality_score,
        max_summary_length_factor=max_summary_length_factor,
        augmentation_loops=augmentation_loops,
    )
    
    dataset = dataset_generator.generate(
            documents=documents
        )
    
    step_context = get_step_context()
    
    step_context.add_output_metadata(
        output_name = "summary_dataset",
        metadata = {
            "count" : len(dataset)
        }
    )
    
    return dataset