from loguru import logger
from we_queue_client import QueueClient, QueueConfig
from .message_handler import InferenceMessageHandlder
from .config import InferenceConfig

if __name__ == "__main__":
    config = InferenceConfig()

    message_handler = InferenceMessageHandlder(config=config)
    logger.info(repr(config))
    QueueClient(config=config,
                message_handler=message_handler).start_listening()
