from loguru import logger
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pydantic based settings class to manage app configuration"""

    # Pydantic Settings
    model_config : SettingsConfigDict = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )
    
    # Notion API Configuration
    NOTION_SECRET_KEY : str | None = Field(
        default=None, description="secret key for Notion API authentication"
    )
    
    # --- AWS Configuration --- 
    AWS_ACCESS_KEY: str | None = Field(
        default=None, description="AWS access key for authentication."
    )
    AWS_SECRET_KEY: str | None = Field(
        default=None, description="AWS secret key for authentication."
    )
    AWS_DEFAULT_REGION: str = Field(
        default="ap-south-1", description="AWS region for cloud services."
    )
    AWS_S3_BUCKET_NAME: str = Field(
        default="demo_projects_data",
        description="Name of the S3 bucket for storing application data.",
    )
    
    # --- MongoDB Atlas Configs --- 
    MONGODB_URI: str = Field(
        default=None, 
        description="Connection URI for the local MongoDB Atlas instance."
    )
    
    MONGODB_DATABASE_NAME: str = Field(
        default="second_brain",
        description="Name of the MongoDB databse."
    )
    
    # --- Huggingface_hub configs ---
    HUGGINGFACE_ACCESS_TOKEN: str = Field(
        default="",
        descrption="authentication token for the huggingface_hub."
    )
    
    HUGGINGFACE_DATASET_ID: str = Field(
        default="Parth19Dhimmar/second_brain_summary_dataset",
        descrption="huggingface summary datset id."
    )
    
    HUGGINGFACE_DEDICATED_ENDPOINT: str = Field(
        default="",
        description="Huggingface summarization model dedicated endpoint for inference"
    )
    
    # --- Gemini / Google AI Studio ---
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="API key for Google Gemini models used by LiteLLM.",
    )
    
    GROQ_API_KEY: str | None = Field(
        default=None,
        description="API key to use Groq models through LiteLLM.",
    )
    
try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to load configuration :  {e}")
    raise SystemExit(e)
    
