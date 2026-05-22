from we_queue_client import MessageHandler, TaskStatus, InternalError, QueueWorkerError
from .message import InferenceInputMessage, InferenceOutputMessage
from .data_processor import DataProcessor
from .storage import InferenceStorage
from .config import InferenceConfig


class InferenceMessageHandlder(MessageHandler):
    def __init__(self, config: InferenceConfig):
        super().__init__(config,
                         InputMessageSchema=InferenceInputMessage,
                         OutputMessageSchema=InferenceOutputMessage)
        self.output_bucket = config.OUTPUT_BUCKET
        self.storage = InferenceStorage(minio_url=config.MINIO_URL,
                                        minio_access_key=,
                                        minio_secret_key=,
                                        aws_https=config.AWS_HTTPS,
                                        vsi_cache_size=config.VSI_CACHE_SIZE)
        # creeate output bucket


    def handle_message(self, message: InferenceInputMessage) -> InferenceOutputMessage:
        data_processor = DataProcessor(pipeline=message.input.pipeline,
                                       input_artifacts=message.input.source_data,
                                       output_artifacts=message.output.output_data,
                                       aoi=message.input.aoi,
                                       storage=self.storage,
                                       enable_blocks=message.input.blocks)
        data_processor.run(task_id=message.task_id,
                           processing_id=message.processing_id,
                           bucket=self.output_bucket)

        return InferenceOutputMessage(task_id=message.task_id,
                                      status=TaskStatus.OK.value)