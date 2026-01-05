from typing import Literal, Union
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

EmbeddingModelType = Literal["openai", "huggingface"] # literal - variable can only take these values
EmbeddingsModel = Union[OpenAIEmbeddings, HuggingFaceEmbeddings] # either of two

def get_embedding_model(
    model_id: str,
    model_type: EmbeddingModelType = "huggingface",
    device: str = "cpu",
) -> EmbeddingsModel:
    """Gets an instance of the configured embedding model.

    The function returns either an OpenAI or HuggingFace embedding model based on the
    provided model type.

    Args:
        model_id (str): The ID/name of the embedding model to use
        model_type (EmbeddingModelType): The type of embedding model to use.
            Must be either "openai" or "huggingface". Defaults to "huggingface"
        device (str): The device to use for the embedding model. Defaults to "cpu"

    Returns:
        EmbeddingsModel: An embedding model instance based on the configuration settings

    Raises:
        ValueError: If model_type is not "openai" or "huggingface"
    """
    
    if model_type == "openai":
        return get_openai_embedding_model(model_id=model_id)
    elif model_type == "huggingface":
        return get_hugggingface_embedding_model(model_id, device=device,)
    else:
        raise ValueError("Provider does not match, enter valid provider.")
    
    
def get_openai_embedding_model(
    model_id: str = "text-embedding-3-large"
) -> OpenAIEmbeddings:
    """Gets an OpenAI embedding model instance.

    Args:
        model_id (str): The ID/name of the OpenAI embedding model to use

    Returns:
        OpenAIEmbeddings: A configured OpenAI embeddings model instance with
            special token handling enabled
    """
    embedding_model = OpenAIEmbeddings(
        model=model_id,
        
    )
    return embedding_model


def get_hugggingface_embedding_model(
    model_id: str,
    device: str,
) -> HuggingFaceEmbeddings:
    """Gets a HuggingFace embedding model instance.

    Args:
        model_id (str): The ID/name of the HuggingFace embedding model to use
        device (str): The compute device to run the model on (e.g. "cpu", "cuda")

    Returns:
        HuggingFaceEmbeddings: A configured HuggingFace embeddings model instance
        with remote code trust enabled and embedding normalization disabled
    """


    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={
            "device": device,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": False,
        },
    )

    logger.info(f"Successfully loaded HF embedding model: {model_id}")

    return embeddings