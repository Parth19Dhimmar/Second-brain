import json
import os 
from pathlib import Path
import random

from pydantic import BaseModel
from loguru import logger
from datasets import Dataset, DatasetDict

class InstructDatasetSample(BaseModel):
    instruction: str
    answer: str
    
class InstructDataset:
    train: list[InstructDatasetSample]
    validation: list[InstructDatasetSample]
    test: list[InstructDatasetSample]   
    test_split_ratio: float = 0.1
    val_split_ratio : float = 0.1
    seed: int | None = None
    
    @classmethod
    def from_sample(
        cls,
        samples: list[InstructDatasetSample],
        val_split_ratio: float,
        test_split_ratio: float,
        seed: int | None = None,
    ) -> "InstructDataset": 
        """Creates an InstructDataset by splitting samples into train/val/test sets.

        Args:
            samples: List of samples to split
            val_split_ratio: Ratio of samples to use for validation (between 0 and 1)
            test_split_ratio: Ratio of samples to use for testing (between 0 and 1)
            seed: Random seed for shuffling. If None, no fixed seed is used.

        Returns:
            InstructDataset with shuffled and split samples
        """
        shuffled_samples = samples.copy()
        
        if seed is not None:
            random.seed(seed)
        
        shuffled_samples = random.shuffle(samples)
        
        train_samples = shuffled_samples[
            : int(len(shuffled_samples) * (1 - val_split_ratio - test_split_ratio))
        ]
        
        val_samples = shuffled_samples[
            int(len(shuffled_samples) * (1 - val_split_ratio - test_split_ratio)) : 
            int(len(shuffled_samples) * (1 - test_split_ratio))
        ] 
        
        test_samples = shuffled_samples[
            int(len(shuffled_samples) * (1 - test_split_ratio)) : 
        ]
        
        logger.info(
           f"Created dataset with the following split:" 
           f" Train_samples : {len(train_samples)}"
           f" Val_samples : {len(val_samples)}"
           f" Test_samples: {len(test_samples)}"
        )
        
        assert len(train_samples) > 0, "Train split must have at least one sample"
        assert len(val_samples) > 0, "Validation split must have at least one sample"
        assert len(test_samples) > 0, "Test split must have at least one sample"
        
        return InstructDataset(
            train=train_samples,
            validation=val_samples,
            test=test_samples,
            val_split_ratio=val_split_ratio,
            test_split_ratio=test_split_ratio,
            seed=seed
    )
        
    
    def to_hf_dataset(self) -> Dataset:
        """Convert to Hugging Face DatasetDict for training or upload."""
        
        return DatasetDict({
            "train": Dataset.from_list([s.model_dump() for s in self.train]),
            "validation": Dataset.from_list([s.model_dump() for s in self.validation]),
            "test": Dataset.from_list([s.model_dump() for s in self.test]),
        })
    
    def write(self, output_dir: Path):
        """Writes the dataset splits to JSON files in the specified directory.

        Args:
            output_dir: Directory path where the dataset files will be saved

        Returns:
            Path to the output directory containing the saved files
        """
        
        train = [s.model_dump() for s in self.train]
        validation = [s.model_dump() for s in self.validation]
        test = [s.model_dump() for s in self.test]
        
        output_dir.mkdir(parents=True, exist_ok=True)

        for sample_name, samples in {
            "train": train,
            "validation": validation,
            "test": test,
        }.items():
            output_path = output_dir + f"{sample_name}.json"
            with open(output_path, "w") as file:
                json.dump(samples, file, indent=2)
            
        logger.info(f"Wrote dataset splits to {output_dir}")

        return output_dir