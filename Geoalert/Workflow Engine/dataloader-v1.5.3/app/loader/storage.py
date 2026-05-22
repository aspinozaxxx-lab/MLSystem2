from boto3 import resource
from botocore.client import Config as Botoconfig
from ..config import Config

class Storage:
    def __init__(self,
                 config: Config):
        """
        Class for working with minio storage, based on boto3
        It uses Artifacts as a representation of files in storage and on the local machine

        Args:
            minio_url: url of minio server
            minio_access_key:  key for minio
            minio_secret_key:  key for minio
        """
        minio_url = config.MINIO_HOST if not config.MINIO_PORT else f"{config.MINIO_HOST}:{config.MINIO_PORT}"
        minio_access_key = 
        minio_secret_key = 
        if not minio_url.startswith('http://') and not minio_url.startswith('https://'):
            minio_url = 'http://' + minio_url

        # import inside the init, so that we will not need boto3 for workers which do not use minio

        self.s3_resource = resource('s3',
                                    endpoint_url=minio_url,
                                    aws_access_key_id=,
                                    aws_secret_access_key=,
                                    config=Botoconfig(signature_version='s3v4'))

    def upload(self, bucket, filename, path):
        """
        Upload artifact from local folder to s3
        Args:
            bucket: remote bucket name
            filename: remote file path (relative to bucket)
            path: local path (str or Path)
        """
        s3_bucket = self.s3_resource.Bucket(name=bucket)
        s3_bucket.upload_file(
            Filename=str(path),
            Key=filename
        )
