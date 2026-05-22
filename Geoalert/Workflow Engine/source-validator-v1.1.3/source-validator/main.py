from loguru import logger
from we_queue_client import QueueClient, QueueConfig
from .message_handler import SourceValidatorMessageHandlder
from .config import SourceValidatorConfig

if __name__ == "__main__":
    config = SourceValidatorConfig()

    message_handler = SourceValidatorMessageHandlder(config=config)
    logger.info(repr(config))
    QueueClient(config=config,
                message_handler=message_handler).start_listening()
