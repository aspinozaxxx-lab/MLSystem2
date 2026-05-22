import os

# tmp - fix bad env param; remove later
def prepare_userinfo_url(url: str):
    if '/user?' in url:
        return url.replace('/user?', '/users/{}?')
    return url

class Config:
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = os.getenv("PORT", "8080")

    SERVICE_ROOT_URL = os.getenv("SERVICE_ROOT_URL", "http://{host}:{port}/rest/rasters".format(host=HOST, port=PORT))
    # ----------------------------------------------------------
    # Authentication server configuration ----------------------
    # ----------------------------------------------------------

    # auth server address:
    AUTH_HOST = os.getenv('AUTH_HOST', 'http://auth_server:8050/auth/')

    # Name for error codes
    SERVICE_NAME = 'data-catalog'
    # Logging setup
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    # ------------------------------------------------------------------------
    # Storage configuration
    # ------------------------------------------------------------------------

    # storage address
    MINIO_HOST = os.getenv('MINIO_HOST', 'localhost')
    MINIO_PORT = os.getenv('MINIO_PORT', '9000')

    # storage credentials
    MINIO_ACCESS_KEY = , 'dataCatalogUser')
    MINIO_SECRET_KEY = , '72sjjhfmmmkasdkjdwidjsafkjfh39938fhasdfSSS@#')

    # bucket details
    MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'data-catalog-bucket')

    # ------------------------------------------------------------------------
    # Database configuration
    # ------------------------------------------------------------------------
    # DB_STRING = os.getenv('DB_STRING', 'sqlite:///./sql_app.db')
    DB_STRING = 'postgresql://{username}:{password}@{host}:{port}/{db_name}'
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = , '1234Qq')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'datacatalogdb')
    DB_SCHEMA = os.getenv('DB_SCHEMA', 'public')
    DB_STRING = DB_STRING.format(username=DB_USER,
                                 password=,
                                 host=DB_HOST,
                                 port=DB_PORT,
                                 db_name=DB_NAME,
                                 #schema_name=DB_SCHEMA,
                                 )

    # ------------------------------------------------------------------------
    # Routes configuration
    # ------------------------------------------------------------------------
    ROUTE_PREFIX = os.getenv('ROUTE_PREFIX', '/rest/rasters')

    # ------------------------------------------------------------------------
    # File handling configs
    # ------------------------------------------------------------------------
    MAX_UPLOAD_FILE_SIZE = int(os.getenv('MAX_UPLOAD_FILE_SIZE', 1_000_000_000))
    MAX_IMAGE_AREA_DEGREES = 10  # to mimic WE check
    MAX_IMAGE_SIZE_PIXELS = int(os.getenv('MAX_IMAGE_SIZE_PIXELS', 30_000))  # to limit raster size
    # ------------------------------------------------------------------------
    # Preview size
    # ------------------------------------------------------------------------
    PREVIEW_SIZE_L = int(os.getenv('PREVIEW_SIZE_L', 1024))
    PREVIEW_SIZE_S = int(os.getenv('PREVIEW_SIZE_S', 256))

    # ------------------------------------------------------------------------
    # Memory limit control switch
    # if MEMORY_LIMIT == 0, no memory limit for users
    # ------------------------------------------------------------------------
    MEMORY_LIMIT = bool(int(os.getenv('MEMORY_LIMIT', False)))

    # ------------------------------------------------------------------------
    # Workflow configurations
    # ------------------------------------------------------------------------
    PATH_TO_WD_YML = os.getenv("PATH_TO_WD_YML", "/code/wd.yaml")
    WORKFLOW_ENGINE_LOGIN = os.getenv("WORKFLOW_ENGINE_LOGIN", "")
    WORKFLOW_ENGINE_PASSWORD = , "")
    WORKFLOW_ENGINE_URL = os.getenv("WORKFLOW_ENGINE_URL", "http://auth_server:8050")
    WORKFLOW_ENGINE_BATCH_SIZE = int(os.getenv("WORKFLOW_ENGINE_BATCH_SIZE", 100))
    RASTER_TILE_SERVER_URL = os.getenv("RASTER_TILE_SERVER_URL", "https://rasters-staging.mapflow.ai")
    SYSTEM_ID = os.getenv("SYSTEM_ID", "data-catalog")
    # Sleep time between WE requests
    SLEEP = 10
    # ------------------------------------------------------------------------
    # Auth configurations
    # ------------------------------------------------------------------------
    PUBLIC_KEY_URL = os.getenv('PUBLIC_KEY_URL',
                               'https://sso.rk.dev.nspd.rosreestr.gov.ru/oauth2/public_keys')
    USER_INFO_URL = prepare_userinfo_url(os.getenv('USER_INFO_URL',
                              'https://rk.dev.nspd.rosreestr.gov.ru/api/actors/v2/users/{}?groups=true&organization=false&userData=false'))
    OIDC_CLIENT = os.getenv('OIDC_CLIENT', 'mapflow')
    # ALLOWED_GROUP_IDS are set in environment as comma-separated string
    # i.e. "061229bd-9e3c-4ec0-ade5-f152e913405f, 31e12d53-420f-4973-b881-d6f7d63201f8"
    USER_GROUP_IDS = [group_id.strip() for group_id in os.getenv("USER_GROUP_IDS", "").split(',')]
    ADMIN_GROUP_IDS = [group_id.strip() for group_id in os.getenv("ADMIN_GROUP_IDS", "").split(',')]
    X_ACTOR_ID = os.getenv("X_ACTOR_ID", "")

    # CORS
    ORIGINS = [origin.strip() for origin in os.getenv('CORS_ALLOWED_ORIGINS', "*").split(',')]


    # ------------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------------

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
