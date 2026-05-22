from loguru import logger
from we_queue_client import QueueClient, QueueConfig
from cog_builder.app.message_handler import CogBuilderMessageHandlder
from cog_builder.config import CogBuilderConfig

if __name__ == "__main__":
    config = CogBuilderConfig()
    message_handler = CogBuilderMessageHandlder(config=config)
    logger.info(repr(config))
    QueueClient(config=config,
                message_handler=message_handler).start_listening()
