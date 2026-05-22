import rasterio
from loguru import logger
from pathlib import Path
from we_queue_client import MessageHandler, TaskStatus, InternalError, QueueWorkerError
from we_queue_client.utils import log_time
from tempfile import TemporaryDirectory
from ..functional.build_cog import CogBuilder
from ..functional.errors import CogInvalidAOI, CogReprojectionError
from .message import CogBuilderInputMessage, CogBuilderOutputMessage
from .storage import CogBuilderStorage
from .errors import InvalidAOI, ReprojectionError
from ..config import CogBuilderConfig


class CogBuilderMessageHandlder(MessageHandler):
    def __init__(self, config: CogBuilderConfig):
        super().__init__(config=config,
                         InputMessageSchema=CogBuilderInputMessage,
                         OutputMessageSchema=CogBuilderOutputMessage)
        self.storage = CogBuilderStorage(minio_url=config.MINIO_URL,
                                        minio_access_key=,
                                        minio_secret_key=,
                                        aws_https=config.AWS_HTTPS)

    def handle_message(self, message: CogBuilderInputMessage) -> CogBuilderOutputMessage:
        input = message.input
        input_artifact = input.input_artifact
        output_artifact = message.output.output_artifact
        with TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            self.storage.get_artifact(artifact=input_artifact,
                                      workdir=workdir,
                                      aoi=input.aoi)
            try:
                log_time(level="INFO",
                         log_kwargs=False,
                         log_args=False)(CogBuilder)(input_ds=workdir/input_artifact.name,
                                                     cog_ds=workdir/output_artifact.name,
                                                     workdir=workdir,
                                                     channels=message.input.channels or None,
                                                     aoi=input.aoi,
                                                     compress=input.compress,
                                                     std_width=input.std_width)
            # raise Queue-ready error message from internal error
            except CogInvalidAOI as e:
                raise InvalidAOI(aoi=e.aoi,
                                 reason=e.reason)
            except CogReprojectionError as e:
                raise ReprojectionError(error_message=e.error_message)

            self.storage.upload(workdir=workdir, artifact=output_artifact)
        return CogBuilderOutputMessage(task_id=message.task_id,
                                       status=TaskStatus.OK.value)

