import pandas as pd
import matplotlib.pyplot as plt
from zenml import step

from second_brain_offline.domain import InstructDataset

@step
def create_histogram(dataset: InstructDataset) -> None:
    """
    Analyzes the dataset by visualizing instruction and answer length distributions.
    """

    # Convert each split to pandas DataFrame
    df_train = dataset["train"].to_pandas()
    df_val = dataset["validation"].to_pandas()
    df_test = dataset["test"].to_pandas()

    # Combine splits with a column to identify them
    df_train["split"] = "train"
    df_val["split"] = "validation"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Compute lengths
    df["instruction_length"] = df["instruction"].apply(len)
    df["answer_length"] = df["answer"].apply(len)
    df["length_ratio"] = df["answer_length"] / (df["instruction_length"] + 1e-6)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(df["instruction_length"], bins=50, alpha=0.6, label="Instruction length")
    plt.hist(df["answer_length"], bins=50, alpha=0.6, label="Answer length")
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Instruction and Answer Length Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot ratio
    plt.figure(figsize=(8, 5))
    plt.hist(df["length_ratio"], bins=40, color="orange", alpha=0.7)
    plt.xlabel("Answer / Instruction Length Ratio")
    plt.ylabel("Frequency")
    plt.title("Length Ratio Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Dataset analysis complete.")
