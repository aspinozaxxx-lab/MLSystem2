import os
from dataclasses import dataclass
from we_queue_client import QueueConfig


@dataclass
class SourceValidatorConfig(QueueConfig):

    # ------------------------------------------------------------------------
    # Storage configuration
    # ------------------------------------------------------------------------

    # storage address
    USE_STORAGE: bool = True
    # If requesting external s3 over HTTPS, set to "YES"
    AWS_HTTPS: str = os.getenv('AWS_HTTPS', 'NO')

    # input and outut queue names
    WORKER_NAME = os.getenv('WORKER_NAME', 'validate-source')


    def __repr__(self):
        return super().__repr__()
