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
    
    USE_HUGGINGFACE_DEDICATED_ENDPOINT: bool = Field(
        default=False,
        description="Whether to use the dedicated endpoint for summarizing responses. If True, we will use the dedicated endpoint instead of OpenAI.",
    )
    
    HUGGINGFACE_DEDICATED_ENDPOINT: str | None = Field(
        default=None,
        description="Dedicated endpoint URL for real-time inference. "
        "If provided, we will use the dedicated endpoint instead of OpenAI. "
        "For example, https://um18v2aeit3f6g1b.eu-west-1.aws.endpoints.huggingface.cloud/v1/, "
        "with /v1 after the endpoint URL.",
    )
    
    # --- Gemini / Google AI Studio ---
    GEMINI_API_KEY: str | None = Field(
        default=None,
        description="API key for Google Gemini models used by LiteLLM.",
    )
    
    GEMINI_MODEL_ID: str | None = Field(
        default=None,
        descroption="Model Id of gemini model to be used(include provider, as using litellm)."
    )
    
    # OPIK configuration (Comet Ml)
    
    COMET_API_KEY : str | None = Field(
        default=None, description="API key for cometml and opik service."
    )
    
    COMET_PROJECT_NAME: str | None = Field(
        default=None, description="project name for cometml and opik tracing."
    )
    
    # --- OpenAI API Configuration ---
    OPENAI_API_KEY: str | None = Field(
        default = None, description="API key for OpenAI service authentication.",
    )
    OPENAI_MODEL_ID: str | None = Field(
        default="gpt-4o-mini", description="Identifier for the OpenAI model to be used."
    )
    
try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to load configuration :  {e}")
    raise SystemExit(e)
    