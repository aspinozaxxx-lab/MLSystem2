# Error codes

## Internal errors

- `INTERNAL_ERROR`: A generic internal error
- `AOI_IMPORT_ERROR`: An error occurred while importing AOI. Might be an invalid geometry, etc.
- `EXTERNAL_SYSTEM_ERROR`: An error occurred during a call to an external system (e.g., WE)

## User input errors

- `BAD_REQUEST`: A generic user input error
- `NOT_FOUND`: An entity was not found
- `FILE_PART_MISSING`: A part containing a file was expected, but is missing.
  See https://github.com/jaydenseric/graphql-multipart-request-spec
- `WORKFLOW_DEF_PARSING_ERROR`: Invalid workflow def yaml
- `LOGIN_TAKEN`: A user with the specified login already exists
- `EMPTY_SELECTION`: The selected AOIs contain no data
- `WD_IN_USE`: This WorkflowDef is in use and can't be deleted
- `TOO_LARGE_PROCESSING`: The user tries to create a processing larger than `LARGE_PROCESSING_AREA` but has no
  permission to do so
- `AREA_LIMIT_EXCEEDED`: The user tries to create a processing, but exceeds the accumulative area limit set for this
  user

## Access errors

- `AUTHENTICATION_ERROR`: Wrong login/password
- `BAD_TOKEN`: Invalid JWT token
- `TOKEN_EXPIRED`: The JWT token has expired
- `ACCESS_DENIED`: The user is either anonymous or trying to take prohibited actions

## Env variables

### MiscConfig

| Name                    | Description                                                  | Default value         |
|-------------------------|--------------------------------------------------------------|-----------------------|
| PORT                    | The port of the HTTP server                                  | 8080                  |
| ORIGIN                  | URL of this system, accessible from the web                  | http://localhost:8080 |
| JWT_KEY                 | JWT key used for coding/decoding auth tokens                 | secretKey             |
| DEFAULT_PARTITION_SIZE  | Max partition size in degrees (both dimensions)              | 10000e-5              |
| SENTINEL_PARTITION_SIZE | Partition size for sentinel source type                      | 2000000e-5            |
| MAX_AOIS_PER_PROCESSING | Limit of AOIs in one Processing for API createAndRun request | 10                    |

### TestEnvConfig

| Name                        | Description                                                                                    | Default value |
|-----------------------------|------------------------------------------------------------------------------------------------|---------------|
| TEST_ENV                    | Mock all external services                                                                     | false         |
| TEST_DATA                   | Migrations to populate DB with test data                                                       | -             |
| MOCK_WE_FAILED_PERCENT      | Determines the probability of receiving a `FAILED` status in response from WE (mocked WE only) | 0             |
| MOCK_WE_IN_PROGRESS_PERCENT | Determines the probability of receiving an `IN_PROGRESS` status                                | 50            |

### DefaultDbConfig

| Name                         | Description                                              | Default value |
|------------------------------|----------------------------------------------------------|---------------|
| DB_PORT                      | Database port                                            | 5432          |
| DB_HOST                      | Database host                                            | database      |
| DB_NAME                      |                                                          | mapflow       |
| DB_USER                      |                                                          | postgres      |
| DB_PASSWORD                  |                                                          |               |
| DB_SCHEMA                    |                                                          | geoalert      |
| CONNECTION_POOL_SIZE         | Maximum number of DB connections                         | 20            | 
| CONNECTION_LEAK_THRESHOLD_MS | Timeout for leaked connections detection im milliseconds | 30000         | 

### AvantpostConfig

| Name              | Description                            | Default value     |
|-------------------|----------------------------------------|-------------------|
| USER_GROUP_IDS    | User group ids, to identify User Role  | [""]              |
| ADMIN_GROUP_IDS   | User group ids, to identify Admin Role | [""]              |
| AVANPOST_URL      |                                        | http://localhost/ |
| AVANPOST_ACTOR_ID |                                        | ""                |

### DefaultWeConfig

| Name                     | Description                                                               | Default value |
|--------------------------|---------------------------------------------------------------------------|---------------|
| SYSTEM_ID                | A string that identifies this app instance, used to track workflow owners | mapflow       |
| MIN_PRIORITY             | Minimum workflow queue priority value                                     | 1             |
| MAX_PRIORITY             | Maximum workflow queue priority value                                     | 10            |
| PROGRESS_UPDATE_INTERVAL | WE polling interval                                                       | 5             |

### DefaultExternalSystemConfig

| Name                    | Description                                   | Default value                                                       |
|-------------------------|-----------------------------------------------|---------------------------------------------------------------------|
| WORKFLOW_ENGINE_URL     |                                               | http://localhost:8060                                               |
| TILEPROXY_URL           | URL of HEAD tiles proxy                       | https://app.mapflow.ai/tiles/satimagery/{z}/{x}/{y}.png?year={year} |
| TILEPROXY_API_KEY       | API key for authenticvation in tileproxy      |                                                                     |
| BILLING_ENGINE_URL      | URL of the Billing Engine                     | http://billing-engine:8080/                                         |
| BILLING_ENGINE_API_KEY  | API key for authenticvation in Billing Engine | ""                                                                  |
| VECTOR_PROCESSOR_URL    |                                               | http://localhost:8700                                               |
| VECTOR_TILE_SERVER_URL  |                                               | http://localhost:8600                                               |
| RASTER_TILE_SERVER_URL  |                                               | http://localhost:8500                                               |
| MINIO_URL               |                                               | http://localhost:9000                                               |
| MINIO_ACCESS_KEY        |                                               | ""                                                                  |
| MINIO_SECRET_KEY        |                                               | ""                                                                  |
| ZOOM_CONSTRAINT         | Max maxar images zoom for non-premium users   | 12                                                                  |
| MAXAR_DISCOVERY_API_KEY | Maxar Discovery API key                       |                                                                     |

### DefaultProcessingReviewConfig

| Name                               | Description                                                          | Default value |
|------------------------------------|----------------------------------------------------------------------|---------------|
| REVIEW_AUTO_CONFIRM_INTERVAL_HOURS | Automatically confirm IN_REVIEW processings after specified interval | 72            |
