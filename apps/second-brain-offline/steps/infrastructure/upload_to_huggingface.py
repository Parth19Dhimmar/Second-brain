from typing import Annotated

from zenml import step, get_step_context
from datasets import Dataset

from second_brain_offline.domain import InstructDataset
from second_brain_offline.config import settings


@step
def upload_to_huggingface(
    dataset: Annotated[InstructDataset, "instruct_dataset"],
    dataset_id: Annotated[str, "dataset_id"]
) -> Annotated[str, "output"]:
    
    hf_token = settings.HUGGINGFACE_ACCESS_TOKEN
    
    assert hf_token is not None, "The huggingface_hub access token is missing"
    
    huggingface_dataset = dataset.to_hf_dataset()
    huggingface_dataset.push_to_hub(dataset_id=dataset_id, token=hf_token)
    
    step_context = get_step_context()
    
    step_context.add_output_metadata(
        output_name="output",
        metadata= {
            "dataset_id": dataset_id,
            "train": huggingface_dataset["train"],
            "validation": huggingface_dataset["validation"],
            "test": huggingface_dataset["test"],    
        }
    )
    
    return dataset_id