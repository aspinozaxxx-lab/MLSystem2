COG Builder
-----------

COG builder is a tool for turning any GeoTIFF file into Cloud-Optimized GeoTIFF (COG)

Supportes GeoTIFF formats: 
1. 3-channels GeoTIFFs with or without `nodata` specified
2. Single-channel (greyscale) GeoTIFFs
3. Multichannel GeoTIFFs (R,G,B channel indices can be specified using `channels` parameter)

If `dtype` of an input GeoTIFF is different from `uint8`, pixel values will be uniformly scaled to fit `uint8`.

If GeoTIFF projection is different from EPSG:3857 (WebMercator) it will be reprojected to EPSG:3857

If GeoTIFF resolution (EPSG:3857) is less then 1 cm/px a GeoTIFF will be rejected because it is most likely caused by invalid Coordinate System 

If GeoTIFF resolution (EPSG:3857) is less then 7.5 cm/px, COG resolution will be limited to 7.5cm or 21 zoom 


Cropping COG to provided geometry
---------------------------------

COG Builder can crop image to provided geometry using optional `mask` parameter. The geometry should be a valid Polygon geometry in terms of GeoJSON

Environment variables
=====================

| Name             | Description                                                      | Default value |
|------------------|------------------------------------------------------------------|---------------|
| AWS_HTTPS        | Use HTTPS for accesssing S3. Internal Minio requires AWS_HTTP=NO | NO            |
| MINIO_HOST       | Minio host                                                       | None          |
| MINIO_PORT       | Minio oirt                                                       | None          |
| MINIO_ACCESS_KEY | Minio access key ID                                              | None          |
| MINIO_SECRET_KEY | Minio access secret                                              | None          |

### Queue config
| Name                  | Description                                | Default value |
|-----------------------|--------------------------------------------|---------------|
| RABBITMQ_HOST         | RabbitMQ host                              | queue         |
| RABBITMQ_NODE_PORT    | RabbitMQ port                              | 5672          |
| RABBITMQ_DEFAULT_USER | RabbitMQ username                          | user          |
| RABBITMQ_DEFAULT_PASS | RabbitMQ password                          | password      |
| RABBITMQ_HEARTBEAT    | RabbitMQ heartbeat interval in seconds     | 7200          |
| RABBITMQ_TIMEOUT      | RabbitMQ timeout                           | 7200          |
| RABBITMQ_MAX_PRIORITY | RabbitMQ maximum task priority             | 10            |
| RETRY_TIMEOUT_SECONDS | timeout before reconnect to queue, seconds | 5             |
| INPUT_QUEUE           | Queue name postfix. Do not change          | .tasks.queue  |
| OUTPUT_QUEUE          | Queue name postfix. Do not change          | .result.queue |

### App config
| Name           | Description                   | Default value    |
|----------------|-------------------------------|------------------|
| WORKER_NAME    | WE Worker name. Do not change | raster-processor |
| LOG_LEVEL      | Logging level                 | INFO             |
| VSI_CACHE_SIZE | GDAL setting                  | 1000000000       |


Installation
============

System requirements:
1. Python >= 3.8
2. GDAL >= 3.3
3. PIP 

Install requirements:
```
python3 -m pip install -r requirements.txt
```
The cog builder itself does not require installation

Unit testing
============

```commandline
python3 -m pytest tests
```

CLI tool 
========
COG Builder provides command line interface useful for ad hoc processings and testing. 
Default compression method is WEBP, you can change it with `--compress <COMPRESSION_TYPE>` option
For methods see https://gdal.org/drivers/raster/cog.html#general-creation-options
```
AWS_S3_ENDPOINT="minio-staging.mapflow.ai" AWS_HTTPS="YES" AWS_ACCESS_KEY_ID=  python3 -m cog_builder.cli s3://workflow-white-maps/workflow-4105224/9edb2f63-f280-47be-94dc-048d136ebcdf/area-4105225.tif s3://workflow-white-maps/workflow-4105224/9edb2f63-f280-47be-94dc-048d136ebcdf/area-4105225_cog.tif --channels=1,2,3 --mask "{\"type\":\"Polygon\",\"coordinates\":[[[61.387278236004995,55.16025082025],[61.38726071074319,55.160251806334685],[61.387243858967366,55.160254726693935],[61.38722832828181,55.16025946909956],[61.387214715522084,55.16026585130295],[61.38720354381889,55.16027362803893],[61.387195242494506,55.160282500451146],[61.38719013056421,55.16029212757717],[61.38718840447658,55.16030213945154],[61.38718840447658,55.183281998574444],[61.38719013056421,55.183292004676474],[61.387195242494506,55.18330162624725],[61.38720354381889,55.1833104935356],[61.387214715522084,55.18331826577722],[61.38722832828181,55.18332464428994],[61.387243858967366,55.18332938395183],[61.38726071074319,55.18333230262091],[61.387278236004995,55.18333328813481],[61.417708096225624,55.18333328813481],[61.41772562148743,55.18333230262091],[61.41774247326325,55.18332938395183],[61.4177580039488,55.18332464428994],[61.417771616708535,55.18331826577722],[61.41778278841173,55.1833104935356],[61.4177910897361,55.18330162624725],[61.4177962016664,55.183292004676474],[61.41779792775404,55.183281998574444],[61.41779792775404,55.16030213945154],[61.4177962016664,55.16029212757717],[61.4177910897361,55.160282500451146],[61.41778278841173,55.16027362803893],[61.417771616708535,55.16026585130295],[61.4177580039488,55.16025946909956],[61.41774247326325,55.160254726693935],[61.41772562148743,55.160251806334685],[61.417708096225624,55.16025082025],[61.387278236004995,55.16025082025]]]}"
```

Note, input and output files could also be a local files:

```
python3 -m cog_builder.cli input.tif cog.tif [--compress <JPEG/LZW/NONE/...>]
```
