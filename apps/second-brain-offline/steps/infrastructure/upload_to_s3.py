import shutil
from pathlib import Path

from typing_extensions import Annotated
from zenml import step, get_step_context

from second_brain_offline import settings
from second_brain_offline.domain import Document
from second_brain_offline.infrastructure.mongo.aws import S3client

@step
def upload_to_s3(
    folder_path: Path,
    s3_prefix: str = ""
    ) -> Annotated[str, "output"]:
    
    s3_client = S3client(bucket_name=settings.AWS_S3_BUCKET_NAME)
    s3_client.upload_folder(folder_path, s3_prefix)
    
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name = "output",
        metadata={
            folder_path : str(folder_path),
            s3_prefix : str(s3_prefix)
        }
    )
    
    return str(folder_path)