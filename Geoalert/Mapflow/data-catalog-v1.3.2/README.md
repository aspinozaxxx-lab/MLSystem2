# data-catalog

Manage the data you uploaded via Mapflow API

## Configuration

### HTTP server config

| Name             | Description               | Default value                       |
|------------------|---------------------------|-------------------------------------|
| HOST             | HTTP listening host       | 127.0.0.1                           |
| PORT             | HTTP listening port       | 8080                                |
| SERVICE_ROOT_URL | Full url                  | "http://{HOST}:{PORT}/rest/rasters" |
| ROUTE_PREFIX     | URL prefix for every path | /rest/rasters                       |

If SERVICE_ROOT_URL is not set, it is defined with default pattern filled with HOST and PORT
If set explicitly, HOST and PORT are not used

### Database config

| Name        | Description       | Default value |
|-------------|-------------------|---------------|
| DB_PORT     | Database port     | 5432          |
| DB_HOST     | Database host     | localhost     |
| DB_NAME     | Database name     | datacatalogdb |
| DB_USER     | Database user     | postgres      |
| DB_PASSWORD | Database password | 1234Qq        |
| DB_SCHEMA   | Database schema   | public        |


### Workflow config

Data-catalog invokes Workflow for building COG for raster tiles serving.
See https://lab.hub.nspd.rosreestr.gov.ru/nspd/geoalert/workflow-engine/workflow-engine

| Name                       | Description                  | Default value                      |
|----------------------------|------------------------------|------------------------------------|
| PATH_TO_WD_YML             | Absolute path of WD          | "/code/wd.yaml"                    |
| WORKFLOW_ENGINE_LOGIN      | Username for Workflow Engine | ""                                 |
| WORKFLOW_ENGINE_PASSWORD   | Password for Workflow Engine | ""                                 |
| WORKFLOW_ENGINE_URL        | URL of Workflow Engine       | http://auth_server:8050            |
| WORKFLOW_ENGINE_BATCH_SIZE | Database port                | 100                                |
| RASTER_TILE_SERVER_URL     | Database host                | https://rasters-staging.mapflow.ai |
| SYSTEM_ID                  | Database host                | data-catalog                       |

### Minio config
Data-catalog stores all the data in Minio bucket

| Name             | Description      | Default value                              |
|------------------|------------------|--------------------------------------------|
| MINIO_PORT       | Minio port       | 9000                                       |
| MINIO_HOST       | Minio host       | localhost                                  |
| MINIO_ACCESS_KEY | Minio access key | dataCatalogUser                            |
| MINIO_SECRET_KEY | Minio secret key | 72sjjhfmmmkasdkjdwidjsafkjfh39938fhasdfSSS |
| MINIO_BUCKET     | Bucket for files | data-catalog-bucket                        |

### File parameters 
| Name                   | Description                              | Default value |
|------------------------|------------------------------------------|---------------|
| MEMORY_LIMIT           | if set to 0, memory limit is not checked | 0             |
| PREVIEW_SIZE_L         | Image preview size (large)               | 1024          |
| PREVIEW_SIZE_S         | Image preview size (small)               | 256           |
| MAX_UPLOAD_FILE_SIZE   | Max single file size, bytes              | 1_000_000_000 |
| MAX_IMAGE_AREA_DEGREES | Max image swath (degrees lat/lon)        | 10            |
| MAX_IMAGE_SIZE_PIXELS  | Max image size in pixels (one side)      | 30_000        |

### Auth config

| Name           | Description                                                     | Default value                                                                                             |
|----------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| PUBLIC_KEY_URL | URL to request public key                                       | https://sso.rk.dev.nspd.rosreestr.gov.ru/oauth2/public_keys                                               |
| USER_INFO_URL  | URL to avanpost actors API                                      | https://rk.dev.nspd.rosreestr.gov.ru/api/actors/v2/users/{}?groups=true&organization=false&userData=false |
| OIDC_CLIENT    | OIDC client name (audience)                                     | mapflow                                                                                                   |
| USER_GROUP_IDS | IDS of groups that are allowed as User. Comma-separated string  | ""                                                                                                        |
| USER_GROUP_IDS | IDS of groups that are allowed as Admin. Comma-separated string | ""                                                                                                        |
| X_ACTOR_ID     | Actor ID for login in actors API                                | ""                                                                                                        |

### Internal config
| Name         | Description                                     | Default value                   |
|--------------|-------------------------------------------------|---------------------------------|
| SERVICE_NAME | Name to add in log messages and error responses | data-catalog                    |
| LOG_LEVEL    | Log level (string - DEBUG, INFO, WARNING etc)   | INFO                            |
| AUTH_URL     | Endpoint to mapflow-api /user/status API         | 'http://auth_server:8050/auth/' |

AUTH_URL is used to get more info about user's limits and permissions, if MEMORY_LIMIT is not 0
It depends on mapflow-api: https://lab.hub.nspd.rosreestr.gov.ru/nspd/geoalert/mapflow/mapflow-api

## Getting started


### Run data-catalog:
   - `$ make run`

### Run tests:

- `$ make test`


### API parameters:


| Resource                                              | Method | Description                                                                                                                                                                             |
|-------------------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| /rest/rasters/mosaic                                  | GET    | Get all mosaics of user. Query parameters: tags (optional, comma-separated list of tags). Tags are optional. In case, if tags are defined, return only mosaics that have specified tags |
| /rest/rasters/mosaic                                  | POST   | Create new mosaic and return mosaic_id. Query parameters: name (required, string), tags (optinal, comma separated list of tags)                                                         |
| /rest/rasters/mosaic/{mosaic_id}                      | GET    | Get mosaic by id. Path parameters: mosaic_id (required, uuid)                                                                                                                           |
| /rest/rasters/mosaic/{mosaic_id}                      | PUT    | Update mosaic by id. Path parameters: mosaic_id (required, uuid). Request body: {"tags": ["string"], "shared": false}                                                                   |
| /rest/rasters/mosaic/{mosaic_id}                      | DELETE | Delete mosaic by id. Path parameters: mosaic_id (required, uuid)                                                                                                                        | 
| /rest/rasters/mosaic/{mosaic_id}/image                | GET    | Get mosaic images. Path parameters: mosaic_id (required, uuid)                                                                                                                          | 
| /rest/rasters/image/{image_id}                        | DELETE | Delete image by id. Path parameters: image_id (required, uuid)                                                                                                                          |
| /rest/rasters/image/{image_id}                        | GET    | Get image by id from mosaic. Path parameters: image_id (required, uuid)                                                                                                                 |
| /rest/rasters/mosaic/image                            | POST   | Create mosaic and upload file to that mosaic. Query parameters: name(required, string), tags (optional, list of tags). Request body: file (required, file being uploaded)               | 
| /rest/rasters/mosaic/{mosaic_id}/link-image           | POST   | Link existing image from minio to mosaic. Path parameters: mosaic_id (required, uuid). Request body: {"url": "link-to-file-at-minio"}                                                   | 
| /rest/rasters/mosaic/{mosaic_id}/image                | POST   | Upload image to existing mosaic. Path parameters: mosaic_id (required, uuid). Request body: file (required, file being uploaded)                                                        |
| /rest/rasters                                         | POST   | Legacy whitemaps API. Create mosaic and upload file into that mosaic. Request body: file (required, file being uploaded)                                                                |
| /rest/rasters/memory                                  | GET    | Get memory stats of user.                                                                                                                                                               | 
| /rest/rasters/image/{image_id}/preview/{preview_type} | GET    | Get image preview by image_id and preview type. Path parameters: image_id (required, uuid), preview_type (required, str, "l" or "s")                                                    |
| /rest/rasters/heartbeat/lite                          | GET    | Healthcheck lite                                                                                                                                                                        |
| /rest/rasters/heartbeat                               | GET    | Healthcheck including DB healthcheck                                                                                                                                                    |



### Data-catalog errors description:
| Error code             | Parameters                                                   | Description                                                                                                                                                            | 
|------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MemoryLimitExceeded    | available_memory, memory_requested                           | You've requested {memory_requested} bytes to upload the file, however you've got {available_memory} of bytes available                                                 |
| FileTooBig             | actual_file_size, max_file_size                              | Max file size allowed to upload is {max_file_size} bytes, got {actual_file_size} bytes instead                                                                         |
| FileCheckFailed        | filename                                                     | {filename} file check failed. Try to upload proper raster                                                                                                              |
| ItemNotFound           | uid, instance_type                                           | Instance {instance_type} with id: {uid} can't be found                                                                                                                 |
| AccessDenied           | uid, user, instance_type                                     | User {user} don't have access to instance {instance_type} with id {uid}                                                                                                |
| FileAlreadyExists      | url                                                          | File from minio: {url},  which is being linked to mosaic, already exists inside mosaic                                                                                 |
| PreviewNotFound        | image_id                                                     | Preview for image {image_id} can't be found                                                                                                                            |
| InvalidLinkToMinio     | object_url                                                   | Invalid image url for minio provided. Url: {object_url}                                                                                                                |
| MinioObjectDoesntExist | object_url                                                   | Object {object_url} doesn't exists at minio                                                                                                                            |
| FileValidationFailed   | mosaic_id, filename, param_name, got_param, expected_param   | File: {filename} can't be uploaded to mosaic: {mosaic_id}. {param_name} - parameter which can't be validated, got_param: {got_param}, expected_param: {expected_param} |


