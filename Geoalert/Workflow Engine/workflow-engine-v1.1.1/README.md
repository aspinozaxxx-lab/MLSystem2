# API

## Starting a new workflow

`POST http://localhost:8060/api/v0/workflows`

```
{
  "areasOfInterest": [{"geometry":{"type":"Polygon","coordinates":[[[-4.27369062318267,15.3583475150155],[-4.27369062318267,15.3729585563541],[-4.26268632875352,15.3689585563541],[-4.26268632875352,15.3583475150155],[-4.27369062318267,15.3583475150155]]]}}],
  "workflowDefinitionId": 1,
  "params": {
    "priority": "5"
  }
}
```

## Starting a new workflow with predefined target layers

`POST http://localhost:8060/api/v0/workflows`

```
{
  "areasOfInterest": [{"geometry":{"type":"Polygon","coordinates":[[[-4.27369062318267,15.3583475150155],[-4.27369062318267,15.3729585563541],[-4.26268632875352,15.3689585563541],[-4.26268632875352,15.3583475150155],[-4.27369062318267,15.3583475150155]]]}}],
  "workflowDefinitionId": 1,
  "params": {
    "priority": "5",
  	"raster-layer-uri": "s3://urban-raster-source/temp-test-1",
  	"vector-layer-id": "e5f24d71-bd76-45df-b84d-2d0ddb2d8af0"
  }
}
```

# Workflow definition spec

## Workflow definition format

Workflow definitions configs use YAML format. Below is an example of WD config with explanation:

```
name: Test                               # Unique WD name
version: 0                               # A consecutive version number of the WD with this name.
                                         # Versioning starts with 0.
                                         # When a workflow is started, it uses the latest WD version.
                                         # When a workflow is restarted, it uses the version that was used
                                         # upon workflow creation.

stages:                                  # List of stages. Essentially these stages form a directed acyclic graph.
  stage1:                                # Name of the stage (arbitrary string)
    description: Stage 1                 # Description of the stage
    action: action1                      # Action name (one of predefined strings, indentifying actions).
                                         # See `Available actions`.
    config:                              # The stage config
      retries: 1                         # Number of retires for this stage. 1 means that
                                         # one extra attempt will be made after the initial attempt fails.
                                         # Optional, defaults to 0.
      retry_interval: 60                 # Time interval between attempts, in seconds. Optional, defaults to 60.
      params:                            # The params for this stage. See `Available actions`
                                         # for the list of params applicable to different actions.
        param1: value                    # Param name and its string value.
        param2: value
  stage2:
    description: Stage 2
    action: action2
    dependsOn:                           # The list of stages that this stage depends on.
                                         # This stage will be started only after stages from this list are comleted.
                                         # If this list is empty, the stage will be started immediately upon creation.
    - stage1
```

## Available actions

### select-source

This stage is used to select the source of raster data to process.

| Parameter name    | Description                                                                                                                                                                                 | Example value           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| auto_confirm      | If true, this stage will be completed automatically and params listed below will be used as the raster source. Otherwise a user input will be required. Currently only `true` is supported. | true                    |
| source_type       | The raster source type. Supported values: `xyz`, `tms`, `wms`, `quadkey`, `local`, `sentinel_l2a`. This param can be overridden by `source_type` workflow param.                                            | xyz                     |
| url               | The raster source URL. This param can be overridden by `url` workflow param.                                                                                                                | http://test/{x}/{y}/{z} |
| zoom              | The zoom level of the map coverage (for `xyz` and `tms` raster source types)                                                                                                                | 18                      |
| projection        | The coordinate reference system                                                                                                                                                             | epsg:3857               |
| target_resolution | Target raster resolution in degrees per pixel (for `wms` raster source type)                                                                                                                | 0.0005                  |
| raster_login      | Login (for sources that require basic auth)                                                                                                                                                 | login                   |
| raster_password   | Login (for sources that require basic auth)                                                                                                                                                 | password                |


### validate-source

This stage is used to validate provided source for acceptance model criteria
See [Source Validator project](https://lab.bftcom.com/nspd/geoalert/workflow-engine/source-validator) 

| Parameter name    | Description                                                                                                                                                                                 | Example value           |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| requirements      | Base64 encoded string containing requirements for source-validator                                                                                                                          |                         | 

### dataloader

This stage is used to download raster from the source.
See [Source Validator project](https://lab.bftcom.com/nspd/geoalert/workflow-engine/dataloader)

| Parameter name | Description                                                                                                                                             | Example value |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| use_cache      | If true, the system will search for any historical tasks with identical configuration. If it finds one, its result will be reused. Defaults to `false`. | false         |
| bucket         | Minio bucket to store results to                                                                                                                        | bucket        |
| ignore_errors  | Supported for `xyz` and `tms` source types. If true, dataloader will skip tiles that it fails to load.                                                  | false         |
| workers        | Supported for `xyz` and `tms` source types. The amount of parallel workers.                                                                             | 8             |
| buffer         | The size of the buffer to extend the AOI geometry. Can be specified both in degrees (e.g. `0.00010`) and meters (e.g. `100m`).                          | 100m          |

### user-input

This stage is used to block the workflow until a user input (in the form of a Minio artifact) is provided.

| Parameter name | Description                                              | Example value            |
| -------------- | -------------------------------------------------------- | ------------------------ |
| bucket         | Minio bucket to store artifacts to                       | bucket                   |
| inputs         | Comma-separated list of file names that are requested    | file1.geojson, file2.yml |
| recipients     | Comma-separated list of emails to request the user input | a@a.ru, b@b.com          |

### build-cog

This stage is used to build Cloud Optimized GeoTIFFs. These can be efficiently served out in tiles on-the-fly.

| Parameter name | Description                                                                                                                                            | Example value |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| bucket         | Minio bucket to store results to                                                                                                                       | bucket        |
| crop_to_mask   | If true, the features will be cropped to the AOI geometry.                                                                                             | true          |
| buffer         | The size of the buffer to extend the AOI geometry for cropping. Can be specified both in degrees (e.g. `0.00001`) and meters (e.g. `100m`).            | 100m          |
| channels       | The channels of multispectral input to be extracted for COG, exactly 3 values in form of comma-separated string with numbers. Default value is "1,2,3" | 3,2,1         |

### inference

This stage is used to process the raster.

| Parameter name | Description                                                          | Example value             |
| -------------- | -------------------------------------------------------------------- | ------------------------- |
| bucket         | Minio bucket to store results to                                     | bucket                    |
| pipeline       | The name of inference pipeline, or base64-encoded pipeline contents. | forest_semantic_v001.yaml |

### import-vector

This stage is used import inference results into the database.

| Parameter name  | Description                                                                                                                                                                                                                          | Example value                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ |
| vector-layer-id | The ID (UUID) of the vector layer to store features. Either an existing or non-existing layer ID can be specified. In the latter case a new layer will be created. This param can be overridden by `vector-layer-id` workflow param. | 9d965fff-0b64-40fe-98ca-c6730f22349e |
| crop_to_mask    | If true, the features will be cropped to the AOI geometry.                                                                                                                                                                           | true                                 |
| buffer          | The size of the buffer to extend the AOI geometry for cropping. Can be specified both in degrees (e.g. `0.00001`) and meters (e.g. `100m`).                                                                                          | 100m                                 |
| merge_strategy  | The merge strategy used to merge features when multiple imports to a single layer are performed. Supported values: `INSTANCE_SEGMENTATION`, `SEMANTIC_SEGMENTATION`, `NONE`.                                                         | INSTANCE_SEGMENTATION                |
| key_properties  | Supported for `SEMANTIC_SEGMENTATION` merge strategy. Comma-separated list of feature attributes, which define whether two overlapping features should be merged. Two feature with different attribute values will not be merged.    | class_id, density                    |

# Env variables

## Database config
| Name                     | Description                                    | Default value   |
|--------------------------|------------------------------------------------|-----------------|
| DATABASE_URI             | database hostname                              | engine-database |
| DATABASE_NAME            | the name of database will be used              | engine_db       |
| DATABASE_USER            | database user                                  | postgres        |
| DATABASE_PASSWORD        | database password                              | 1234Qq          |
| DATABASE_SCHEMA          | database schema name                           | public          |

### Minio config
| Name             | Description                   | Default value         |
|------------------|-------------------------------|-----------------------|
| minio.location   | minio full location           | http://localhost:9000 |
| minio.host       | minio host address (not used) | localhost             |
| minio.port       | minio port         (not used) | 9000                  |
| minio.access.key | minio access key              | -                     |
| minio.secret.key | minio secret key              | -                     |


### Queue config
| Name                     | Description                                    | Default value   |
|--------------------------|------------------------------------------------|-----------------|
| rabbitmq.host | hours before user-input stage fails by timeout | 72              |
| rabbitmq.node.port | hours before user-input stage fails by timeout | 72              |
| rabbitmq.default.user | hours before user-input stage fails by timeout | 72              |
| rabbitmq.default.pass | hours before user-input stage fails by timeout | 72              |
| rabbitmq.max.priority | hours before user-input stage fails by timeout | 72              |

### App config
| Name                     | Description                                             | Default value                        |
|--------------------------|---------------------------------------------------------|--------------------------------------|
| LOG_LEVEL                | Log level string (DEBUG, INFO etc.)                     | INFO                                 |
| MAIL_HOST                | email server host                                       | smtp.gmail.com                       |
| MAIL_PORT                | email server port                                       | 25                                   |
| MAIL_USERNAME            | email server username                                   | -                                    |
| MAIL_PASSWORD            | email server password                                   | -                                    |
| WORKFLOW_TIMEOUT_HOURS   | hours before workflow fails by timeout                  | 24                                   |
| USER_INPUT_TIMEOUT_HOURS | hours before user-input stage fails by timeout          | 72                                   |
| external.url             | How the service is available from outside               | http://localhost:8060                |
| cors.allowed.origins     |                                                         | http://localhost:3000                |
| runcheck.url             | URL for queue workers to check if the task is cancelled | http://engine.workflow-duty.svc:8080 |
| max.aoi.area.degrees     | Area of interest swath limit, degrees lat/lon           | 10                                   |


# Running Workflow Engine locally

You'll need docker-compose 

1. Run required services locally using docker-compose:
`docker-compose up`
2. Run the application

3. If you need to switch to the local PostgreSQL DB, set environment variable:

```
DATABASE_URI=localhost:5432;
```