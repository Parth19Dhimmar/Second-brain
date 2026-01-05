import os
from pathlib import Path
from typing import Union
import tempfile
import zipfile

import boto3
import botocore
from botocore.exceptions import ClientError
from loguru import logger


from second_brain_offline import settings

class S3client():
    def __init__(
        self,
        bucket_name: str,
        region: str = settings.AWS_DEFAULT_REGION,
        unsigned_request: bool = False # when true allow unauthenticated requests to public s3
    ) -> None:
        
        """Initialize S3 client and bucket name.

        Args:
            bucket_name (str): Name of the S3 bucket
            unsigned_request (bool, optional): If True will access S3 un-authenticated for public buckets.
                If False will use the AWS credentials set by the user. Defaults to False.
            region (str, optional): AWS region. Defaults to AWS_DEFAULT_REGION or AWS_REGION env var,
                or 'ap-south-1'.
        """
        
        self.region = region
        self.bucket_name = bucket_name
        self.unsigned_request = unsigned_request
        
        if self.unsigned_request:
            self.s3client = boto3.client(
                "s3",
                region_name=self.region,
                config=botocore.config.Config(signature_version=botocore.UNSIGNED)
            )
        self.s3client = boto3.client(bucket_name)
        
    def upload_folder(self, local_path: Union[str, Path], s3_prefix: str = "") -> None:
        """_summary_

        Args:
            local_path (Union[str, Path]): _description_
            s3_prefix (str, optional): _description_. Defaults to "".

        Raises:
            FileNotFoundError: _description_
            NotADirectoryError: _description_
        """
        
        # ensure bucket exists before proceeding
        
        self.__create_bucket_if_doesnt_exist()
        
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        if not local_path.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {local_path}")
        
        logger.info(f"Zipping and uploading folder: {local_path} -> s3://{self.bucket_name}/{s3_prefix}")
        
        try:
            # Create a temporary zip file
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
                with zipfile.ZipFile(temp_zip.name, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Walk through all files in the directory
                    for root, _, files in os.walk(local_path):
                        for filename in files:
                            file_path = Path(root) / filename
                            # Add file to zip with relative path
                            zipf.write(file_path, file_path.relative_to(local_path))

                # Construct S3 key with prefix
                zip_filename = f"{local_path.name}.zip"
                s3_key = f"{s3_prefix.rstrip('/')}/{zip_filename}".lstrip("/")

                # Upload zip file
                logger.debug(
                    f"Uploading {local_path} to {self.bucket_name} with key {s3_key}"
                )
                self.s3_client.upload_file(temp_zip.name, self.bucket_name, s3_key)

        except botocore.exceptions.BotoCoreError as e:
            logger.error(f"Failed to upload zip to S3: {e}")
            raise

        finally:
            if temp_zip.name and os.path.exists(temp_zip.name):
                os.unlink(temp_zip.name)
                
    
    def __create_bucket_if_doesnt_exist(self) -> None :
        """Check if bucket exists and create it if it doesn't.
 
        Raises:
            Exception: If bucket creation fails or if user lacks necessary permissions
        """
        try: 
            self.client.head_bucket(
                Bucket= self.bucket_name
            )
        except self.s3client.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "404":
                try:
                    self.s3client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration=self.region
                    )
                except self.s3client.exceptions.ClientError as create_error:
                    raise Exception(
                        f"Failed to create bucket {self.bucket_name} in {self.region} region : {str(create_error)}"
                    )
            elif error_code == "403":
                raise Exception(
                    f"No Permission access to bucket {self.bucket_name}"
                )
            else:
                raise

    
    def download_folder(self, s3_prefix: str, local_path: Union[str, Path]) -> None:
        """Download a zipped folder from S3 and extract it to local storage.

        Args:
            s3_prefix (str): Prefix (folder path) in S3 bucket pointing to the zip file
            local_path (Union[str, Path]): Local path where files should be extracted
        """
        
        local_path = Path(local_path)
        
        # create a temporary file to store zip 
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip:
            # download the zip file
            self.s3_client.download_file(
                self.bucket_name,
                s3_prefix,
                temp_zip.name
            )
            
            # create local dir if it doesn't exist
            local_path.mkdir(exist_ok=True, parents=True)
            
            # extract the zip file
            with zipfile.ZipFile(temp_zip.name, "w") as zipf:
                zipf.extractall(local_path)
                
        # clean up the temp zip file
        os.unlink(temp_zip.name)
        
        
    def download_file(self, s3_prefix: str, local_path: Union[str, Path]) -> None:
        """Download a file from S3 to local storage.

        Args:
            s3_prefix (str): Path to the file in S3 bucket
            local_path (Union[str, Path]): Local directory path where the file should be downloaded
        """

        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)

        target_file = local_path / Path(s3_prefix).name
        self.s3_client.download_file(self.bucket_name, s3_prefix, str(target_file))
                