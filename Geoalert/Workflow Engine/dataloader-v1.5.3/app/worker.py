import sys
from kombu import Connection, Exchange, Queue
from loguru import logger
from time import sleep

from .config import config
from .loader.queue import MessageHandler
from .loader.storage import Storage

RETRY_TIME=10


def start_service():
    # create connection to queue
    conn = Connection(hostname=config.QUEUE_HOST,
                      port=config.QUEUE_PORT,
                      userid=config.QUEUE_USER,
                      password=,
                      heartbeat=config.QUEUE_HEARTBEAT,
                      connect_timeout=config.QUEUE_TIMEOUT)

    exchange = Exchange("test-exchange", type="direct")
    tasks_queue_args = {'x-max-priority': int(config.QUEUE_MAX_PRIORITY)}
    input_queue = Queue(name=config.INPUT_QUEUE,
                        exchange=exchange,
                        durable=True,
                        queue_arguments=tasks_queue_args,
                        routing_key=config.INPUT_QUEUE)
    output_queue = Queue(name=config.OUTPUT_QUEUE,
                         exchange=exchange,
                         durable=True,
                         routing_key=config.OUTPUT_QUEUE)

    # should we declare prefetch here?
    # channel.basic_qos(prefetch_count=1)
    return conn, input_queue, output_queue


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=config.LOG_LEVEL, format="{time:YYYY-MM-DD-HH:mm:ss} {level} {message}")
    storage = Storage(config)
    while True:
        try:
            connection, input_queue, output_queue = start_service()
            break
        except Exception as e:
            logger.info(f'Exception while starting service. Waiting {RETRY_TIME} before retry')
            logger.opt(exception=True).debug("Error details: ")
            sleep(RETRY_TIME)

    logger.info('[*] Waiting for tasks. To exit press CTRL+C')
    logger.info(str(config))
    try:
        worker = MessageHandler(connection=connection,
                                input_queue=input_queue,
                                output_queue=output_queue,
                                storage=storage,
                                config=config)
        worker.run()
    except Exception as e:
        connection.close()
        logger.exception("Closing the connection due to error")

