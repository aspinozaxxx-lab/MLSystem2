Raster Tile Server
==================

Raster Tile Server serves tiles from COGs stored in MINIO

Listen API
----------

HTTP TMS for serving tiles. 

API requests:

### Serve single tile 

`/api/v0/cogs/tiles/18/184380/74226.png?uri=s3://public/urban/default_training_dataset`,
where:
1. `18/184380/74226` tile zoom/x/y 
2. `.png` tile format, only png is supported
3. `uri` COG URI in Minio data storage

### Get resource metadata
`/api/v0/cogs/tiles.json?uri=s3://healthcheck`

Response is self-explanatory:
```
{
    "bounds": [47.9674673079809,42.41012607136868,47.9696881769895,42.41694214088627],
    "center": [47.968577742485195,42.41353410612747],
    "maxzoom": 19,
    "minzoom": 0, 
    "name": "s3://healthcheck",
    "tiles":["https://rasters-duty.mapflow.ai/api/v0/cogs/tiles/{z}/{x}/{y}.png?uri=s3://healthcheck"]
}
```

### Get resource bounds
`/api/v0/cogs/bounds?uri=s3://healthcheck`

Return resource bounds in WebMercator projection.

```
{
    "bounds": [5339714.03536743, 5222613.702401437, 5339961.261374587, 5223641.422107515]
}
```

## Environment variables

### Network config

| Name         | Description                        | Default value         |
|--------------|------------------------------------|-----------------------| 
| PORT         | Listen port                        | 8080                  |
| EXTERNAL_URL | External web URL for user requests | http://localhost:8500 |

### Minio config

| Name             | Description                 | Default value |
|------------------|-----------------------------|---------------| 
| MINIO_HOST       | Minio host name             | localhost     |
| MINIO_PORT       | Minio port                  | 9000          |
| MINIO_ACCESS_KEY | Minio access key (user)     | _mandatory_   |
| MINIO_SECRET_KEY | Minio secret key (password) | _mandatory_   |

### Service config

| Name                                  | Description                                                                                         | Default value |
|---------------------------------------|-----------------------------------------------------------------------------------------------------|---------------| 
| MAX_SOURCES_PER_TILE                  | Max number of images to be read for one tile                                                        | "8"           |
| VSI_CACHE_SIZE                        | GDAL settings                                                                                       | "250000000"   |
| CPL_VSIL_CURL_CACHE_SIZE              | GDAL settings                                                                                       | "250000000"   |
| GDAL                                  | Use JNI GDAL binding. If false Geotrellis pure Java implementation will be used. Recommended = True | TRUE          |
| GDAL_THREAD_POOL_SIZE                 | Minio secret key (password)                                                                         | "50"          |
| ATTRIBUTE_STORE_CACHE_EVICTION_PERIOD | Period in seconds after which attribute store cache will check whether the data is up to date       | "120"         |


  


