import os


class Config:

    MINIO_ACCESS_KEY = , 'Q3AM3UQ867SPQQA43P2F')
    MINIO_SECRET_KEY = , 'zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG')

    MINIO_HOST = os.getenv('MINIO_HOST', 'play.minio.io')
    MINIO_PORT = os.getenv('MINIO_PORT', '9000')

    MINIO_PART_SIZE = 100 * 1024 * 1024

    # messages queue configuration
    QUEUE_HOST = os.getenv('RABBITMQ_HOST', 'queue')
    QUEUE_PORT = os.getenv('RABBITMQ_NODE_PORT', '5672')

    QUEUE_USER = os.getenv('RABBITMQ_DEFAULT_USER', 'user')
    QUEUE_PASSWORD = , 'password')

    QUEUE_HEARTBEAT = int(os.getenv('RABBITMQ_HEARTBEAT', '7200'))
    QUEUE_TIMEOUT = int(os.getenv('RABBITMQ_TIMEOUT', '7200'))

    WORKER_NAME = os.getenv('WORKER_NAME', 'dataloader')
    INPUT_QUEUE = WORKER_NAME + os.getenv('INPUT_QUEUE', '.tasks.queue')
    OUTPUT_QUEUE = WORKER_NAME + os.getenv('OUTPUT_QUEUE', '.result.queue')

    QUEUE_MAX_PRIORITY = os.getenv('RABBITMQ_MAX_PRIORITY', '10')

    HTTP_PROXY = os.getenv('HTTP_PROXY', '')
    CONNECTION_POOL_LIMIT = int(os.getenv('CONNECTION_POOL_LIMIT', '100'))
    # configure default dataloader
    LOADER_IGNORE_ERRORS = os.getenv('LOADER_IGNORE_ERRORS', "false").lower() == "true"
    FAIL_ON_RUNCHECK_ERROR = (os.getenv('READ_OUTPUT_QUEUE', "True")).lower() == "true"

    NSPD_STORAGE_URL = os.getenv('NSPD_STORAGE_URL', 'http://rasters-controller-api.service.consul:5150/v2/rasters/{image_id}')
    NSPD_STORAGE_TIMEOUT = int(os.getenv('NSPD_STORAGE_TIMEOUT', 300))
    # optional header to have access to the rosreestr web maps
    X_ACTOR_ID = os.getenv("X_ACTOR_ID", None)

    DEFAULT_XYZ_LOADER_KWARGS = dict(
        source_type='xyz',
        projection='epsg:3857',
        header={"x-actor-id": X_ACTOR_ID} if X_ACTOR_ID else None,
        credentials=None,
        retry_attempts=5,
        retry_delay=1,
        response_timeout=10,
        rotate_agents=False,
        tile_size=256,
        delete_tiles=True,
        ignore_errors=LOADER_IGNORE_ERRORS,
        connection_limit=CONNECTION_POOL_LIMIT
    )
    # Same as xyz with other source_type
    DEFAULT_TMS_LOADER_KWARGS = dict(
        source_type='tms',
        projection='epsg:3857',
        header={"x-actor-id": X_ACTOR_ID} if X_ACTOR_ID else None,
        credentials=None,
        retry_attempts=5,
        retry_delay=1,
        response_timeout=10,
        rotate_agents=False,
        tile_size=256,
        delete_tiles=True,
        ignore_errors=LOADER_IGNORE_ERRORS,
        connection_limit=CONNECTION_POOL_LIMIT
    )

    # same as XYZ with other source_type
    DEFAULT_QUADKEY_LOADER_KWARGS = dict(
        source_type='quadkey',
        projection='epsg:3857',
        header={"x-actor-id": X_ACTOR_ID} if X_ACTOR_ID else None,
        credentials=None,
        retry_attempts=5,
        retry_delay=1,
        response_timeout=10,
        rotate_agents=False,
        tile_size=256,
        delete_tiles=True,
        ignore_errors=LOADER_IGNORE_ERRORS,
        connection_limit=CONNECTION_POOL_LIMIT)

    DEFAULT_NSPD_LOADER_KWARGS = dict(
        header={"x-actor-id": X_ACTOR_ID} if X_ACTOR_ID else None,
        server=NSPD_STORAGE_URL,
        timeout=NSPD_STORAGE_TIMEOUT
    )


    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
    if LOG_LEVEL not in ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        LOG_LEVEL = "DEBUG"

    # memory limit - in MB. 20000 < 20 GiB, which is the kubernetes limit at the time
    MEMORY_LIMIT = 20000


    def __repr__(self):

        header = "\n\n" + "#" * 30 + " " * (10 - 3) + "CONFIG" + " " * (10 - 3) + "#" * 30 + "\n\n"
        body = ""
        for k, v in self.__class__.__dict__.items():
            if not callable(k) and not k.startswith("_"):
                body += "\t"
                body += f"{k}: {v}\n"
        footer = "\n" + "#" * 80 + "\n"
        return header + body + footer

    def __str__(self):
        return self.__repr__()


config = Config()

# data storage configuration
