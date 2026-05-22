import os
from dataclasses import dataclass
from we_queue_client import QueueConfig


@dataclass
class InferenceConfig(QueueConfig):
    # override: we should fail on startup if the params are not provided
    USE_STORAGE: bool = True
    # If requesting external s3 over HTTPS, set to "YES"
    AWS_HTTPS: str = os.getenv('AWS_HTTPS', 'NO')
    # cache size for reading from minio. If the whole file fits into it, the reading is much faster.
    VSI_CACHE_SIZE: int = int(os.getenv("VSI_CACHE_SIZE", '1_000_000_000'))
    # bucket name for arbitrary outputs
    OUTPUT_BUCKET: str = os.getenv('OUTPUT_BUCKET', 'inference-artifacts')

    # Somehow without this parent repr is not called
    def __repr__(self):
        return super().__repr__()

    def __post_init__(self):
        self.set_gdal_env()

    def set_gdal_env(self):
        """
        Some of the settings need to be in env (for GDAL to read them)
        so we want to set them to reasonable defaults in case they are not specified on launch
        """
        if 'VSI_CACHE' not in os.environ:
            os.environ['VSI_CACHE'] = 'YES'
        if 'VSI_CACHE_SIZE' not in os.environ:
            os.environ['VSI_CACHE_SIZE'] = str(self.VSI_CACHE_SIZE)
        if 'GDAL_DISABLE_READDIR_ON_OPEN' not in os.environ:
            os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = "EMPTY_DIR"
        if 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS' not in os.environ:
            os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = ".tif"
        if 'AWS_HTTPS' not in os.environ:
            os.environ['AWS_HTTPS'] = self.AWS_HTTPS
        if 'AWS_ACCESS_KEY_ID' not in os.environ:
            os.environ['AWS_ACCESS_KEY_ID'] = self.MINIO_ACCESS_KEY
        if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
            os.environ['AWS_SECRET_ACCESS_KEY'] = self.MINIO_SECRET_KEY
        if 'AWS_VIRTUAL_HOSTING' not in os.environ:
            os.environ['AWS_VIRTUAL_HOSTING'] = "NO"
        if 'AWS_S3_ENDPOINT' not in os.environ:
            os.environ['AWS_S3_ENDPOINT'] = ':'.join([self.MINIO_HOST, self.MINIO_PORT]) if self.MINIO_PORT else self.MINIO_HOST


config = InferenceConfig()
