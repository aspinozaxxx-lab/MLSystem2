from loguru import logger
from we_queue_client import MessageHandler, TaskStatus, InternalError, QueueWorkerError

from data_validator_lib import Status as ValStatus
from data_validator_lib.validate import validate
from data_validator_lib.base.error_message import InternalError

from .message import SourceValidatorInputMessage, SourceValidatorOutputMessage
from .storage import SourceValidatorStorage
from .config import SourceValidatorConfig


class SourceValidatorMessageHandlder(MessageHandler):
    def __init__(self, config: SourceValidatorConfig):
        super().__init__(config,
                         InputMessageSchema=SourceValidatorInputMessage,
                         OutputMessageSchema=SourceValidatorOutputMessage)
        self.storage = SourceValidatorStorage(minio_url=config.MINIO_URL,
                                        minio_access_key=,
                                        minio_secret_key=,
                                        aws_https=config.AWS_HTTPS)

    def handle_message(self, message: SourceValidatorInputMessage) -> SourceValidatorOutputMessage:
        requirements = message.input.requirements
        request = message.input.request
        val_status, error_messages = validate(wd=requirements,
                                              request=request,
                                              storage=self.storage)
        status = TaskStatus.FAIL if val_status == ValStatus.ERROR else TaskStatus.OK
        logger.debug(f"Error status from validation is {val_status}, queue status is {status}")
        return SourceValidatorOutputMessage(task_id=message.task_id,
                                            status=status,
                                            messages=[message.schema() for message in error_messages])
