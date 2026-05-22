import os
from tempfile import TemporaryDirectory

from kombu.mixins import ConsumerProducerMixin
from kombu import Connection, Queue
import copy
import json
import shutil
import aiohttp
import asyncio
from loguru import logger
from queue import Queue as qq
from threading import Thread

import maploader
import nspd_loader

from .storage import Storage
from .errors import UnknownSourceType, InternalError, MemoryLimitExceeded, LoaderArgsError
from .message import OutputMessage
from maploader.errors import MaploaderError
from ..config import Config


class MessageHandler(ConsumerProducerMixin):
    def __init__(self,
                 connection: Connection,
                 input_queue: Queue,
                 output_queue: Queue,
                 storage: Storage,
                 config: Config):
        self.connection = connection
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.storage = storage
        self.config = config
        self.queue = qq()
        Thread(target=self.run_tasks).start()

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=[self.input_queue],
                         callbacks=[self.on_message])]

    def run_tasks(self):
        while True:
            try:
                self.on_task(*self.queue.get())
            except Exception as ex:
                logger.exception(ex)
            except KeyboardInterrupt:
                break

    def on_message(self, body, message):
        self.queue.put((body, message))

    def send_message(self, message):
        body = json.dumps(message)
        logger.info(f'Replying with message {message}')
        self.producer.publish(body,
                              exchange='test-exchange',
                              routing_key=self.config.OUTPUT_QUEUE,
                              retry=True,
                              declare=[self.output_queue],
        )

    def on_task(self, body, message):
        task_id = None
        loader_kwargs = None

        try:
            # get input params
            input_kwargs = body
            # Assuming that input will have 'input and 'input' will have 'source_type'
            # https://gitlab.com/geoalert-projects/workflow-engine-project/workflow-engine/
            # -/blob/master/src/main/java/ru/skoltech/aeronetlab/urban/action/selectsource/SelectSourceStageStarter.java#L49
            # input_kwargs['input']['source_type'] = str(input_kwargs['input'].get('source_type').lower())

            input_kwargs = lower_source_type_input_kwargs(input_kwargs)
            task_id = input_kwargs['task_id']
            status_endpoint = input_kwargs.get('runcheck_url', '')

            try:
                is_active = asyncio.run(self.retrying_runcheck(status_endpoint))
            except Exception as e:
                # Means that WE is unavailable
                # current choice is to run the processing in this case
                # however I left the warning here to know what is going on
                if self.config.FAIL_ON_RUNCHECK_ERROR:
                    logger.warning(f'Runcheck endpoint status failure: {str(e)}. '
                                   f'Rejecting task {task_id}')
                    raise e
                else:
                    logger.exception(f'Runcheck endpoint status failure: {str(e)}. Accepting task {task_id} anyway')
                    is_active = True
            else:
                task_status_str = 'active' if is_active else 'cancelled'
                logger.info(f'Runcheck: task {task_id} is {task_status_str}')

            if not is_active:
                # status returned as 'canceled'
                # so we drop all further processing and just send ack (see finally section)
                return

            with TemporaryDirectory() as workdir_name:
                loader, loader_kwargs = self.create_loader_setup(workdir_name, input_kwargs)
                self.download_raster(loader, **loader_kwargs)
                logger.debug('Downloaded to {}'.format(loader_kwargs['output_fp']))
                self.storage.upload(
                    bucket=input_kwargs['output']['bucket'],
                    filename=input_kwargs['output']['filename'],
                    path=loader_kwargs['output_fp'],
                )


            msg = OutputMessage(
                task_id=task_id,
                status=OutputMessage.SUCCESS,
                error_messages=[])
            self.send_message(msg.get())
        except MaploaderError as e:
            logger.exception(f"Replying with status: FAIL, message: {e}")
            msg = OutputMessage(
                task_id=task_id,
                status=OutputMessage.FAIL,
                error_messages=[e.asdict()]
            )
            self.send_message(msg.get())
        except Exception as e:
            logger.exception(f"Replying with status: FAIL, message: {e}")
            msg = OutputMessage(
                task_id=task_id,
                status=OutputMessage.FAIL,
                error_messages=[InternalError(error_message=str(e)).asdict()],
            )
            self.send_message(msg.get())
        finally:
            try:
                message.ack()
            except Exception as e:
                logger.info("Not acked!")

    def create_loader_setup(self, workdir: str, input_kwargs: dict):
        logger.info('Got message with params: {}'.format(input_kwargs))
        source_type = input_kwargs['input']['source_type']

        # get loader
        loaders = {'xyz': (maploader, self.config.DEFAULT_XYZ_LOADER_KWARGS),
                   'tms': (maploader, self.config.DEFAULT_TMS_LOADER_KWARGS),
                   'quadkey': (maploader, self.config.DEFAULT_QUADKEY_LOADER_KWARGS),
                   'nspd': (nspd_loader, self.config.DEFAULT_NSPD_LOADER_KWARGS)}

        # create params for dataloader
        loader_setup = loaders.get(source_type, None)
        if loader_setup is None:
            raise UnknownSourceType(allowed_source_types=list(loaders.keys()), real_source_type=source_type)

        loader = loader_setup[0]
        # make a deep copy in order not to modify config values
        loader_kwargs = copy.deepcopy(loader_setup[1])
        loader_kwargs.update(input_kwargs['input'])
        loader_kwargs['output_fp'] = os.path.join(workdir,
                                                  input_kwargs['output']['filename'])
        return loader, loader_kwargs

    def download_raster(self, loader, **kwargs):
        dtypes = {
            'zoom': int,
            'delay': float,
            'workers': int,
            'retry_attempts': int,
            'retry_delay': float,
            'response_timeout': float,
            'tile_size': int,
            'credentials': tuple,
            'target_resolution': float,
            'connection_limit': int
        }

        # cast strings to appropriate data types
        for k, v in kwargs.items():
            dtype = dtypes.get(k, None)
            if dtype is not None and v is not None:
                try:
                    kwargs[k] = dtype(v)
                except ValueError as e:
                    logger.error('Invalid value for argument {}: {}.'.format(k, v))
                    raise LoaderArgsError(argument_name=k, argument_type=type(v), expected_type=dtype)

        if "estimate_memory" in dir(loader):
            # this function may not be implemented in the loader, in this case we skip the check
            memory_size = loader.estimate_memory(**kwargs)
            logger.debug(f"Estimated RAM usage: {memory_size}")
            if memory_size > self.config.MEMORY_LIMIT:
                raise MemoryLimitExceeded(allowed_size=self.config.MEMORY_LIMIT, estimated_size=memory_size)
        loader.download(**kwargs)

    @staticmethod
    async def check_status(status_endpoint):
        async with aiohttp.ClientSession() as session:
            async with session.get(status_endpoint,
                                   timeout=10,
                                   proxy=None,
                                   raise_for_status=True) as response:
                result = await response.read()

                if int(result) == 0:
                    return True
                elif int(result) == 1:
                    return False
                else:
                    raise ValueError(f"Unexpected response from runcheck endpoint: {result}")

    @staticmethod
    async def retrying_runcheck(status_endpoint, retries=1, sleep_seconds=60):
        for i in range(retries + 1):
            try:
                res = await MessageHandler.check_status(status_endpoint)
                return res
            except Exception as e:
                if i < retries:
                    logger.debug(f"Runcheck failed: {e}, retrying")
                    await asyncio.sleep(sleep_seconds)
                else:
                    raise e

def lower_source_type_input_kwargs(input_kwargs: dict) -> dict:
    input_kwargs['input']['source_type'] = str(input_kwargs['input'].get('source_type').lower())
    return input_kwargs




