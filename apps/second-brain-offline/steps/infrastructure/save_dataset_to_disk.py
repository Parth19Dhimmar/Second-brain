import shutil
from pathlib import Path
from loguru import logger

from typing_extensions import Annotated
from zenml import step, get_step_context

from second_brain_offline.domain import InstructDataset

@step
def save_dataset_to_disk(
    dataset: Annotated[InstructDataset, "instruct_dataset"],
    output_dir: Path
) -> Annotated[str, "output"]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    logger.info(f"Saving dataset to '{output_dir}'")
    output_dir = dataset.write(output_dir)
    
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="output",
        metadata={
            "output_dir": str(output_dir),
            "train_samples": len(dataset.train),
            "test_samples": len(dataset.test),
            "validation_samples": len(dataset.validation),
        }
    )
    
    return str(output_dir)

    
    
    