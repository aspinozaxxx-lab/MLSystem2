# urban-dataloader

Application for loading external raster data. 

Exmaple of test task:
```json
{
  "task_id": 1,
  "input": {
    "url": "https://sat04.maps.yandex.net/tiles?l=sat&v=3.356.0&v=3.261.0&x={x}&y={y}&z={z}&scale=1&lang=ru_RU",
    "geometry": {
      "type": "Polygon",
      "coordinates": [
        [
          [
            41.431310176849365,
            52.730925511477004
          ],
          [
            41.43546223640441,
            52.730925511477004
          ],
          [
            41.43546223640441,
            52.73359567317238
          ],
          [
            41.431310176849365,
            52.73359567317238
          ],
          [
            41.431310176849365,
            52.730925511477004
          ]
        ]
      ]
    },
    "zoom": 16,
    "header": {"KEY-": "-VALUE_TO_ADD_TO_TILE_REQUEST_HEADER"}
  },
  "output": {
    "bucket": "00test",
    "filename": "a_test_file.tif"
  }
}
```

# Tests
```bash
docker build -f tests/docker/Dockerfile -t dataloader:test .
docker run -it --rm dataloader:test pytest tests/
```

# Interactive tests (without rebuilding docker):
```bash
docker build -f tests/docker/Dockerfile.dev -t dataloader:devtest .
docker run -it --rm -v /home/trekin/projects/workflow-engine-project/dataloader/dataloader/:/tests/dataloader -v /home/trekin/projects/workflow-engine-project/dataloader/tests/:/tests dataloader:devtest bash
>>pytest tests/
```

# Integration tests
```bash
cd tests/it
docker compose up
```

# Deployment

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
| Name                  | Description                                                    | Default value                                                             |
|-----------------------|----------------------------------------------------------------|---------------------------------------------------------------------------|
| WORKER_NAME           | WE Worker name. Do not change                                  | dataloader                                                                |
| LOG_LEVEL             | Logging level                                                  | INFO                                                                      |
| HTTP_PROXY            | Proxy (not used - only for Maxar imagery)                      | ""                                                                        |
| CONNECTION_POOL_LIMIT | Simultaneous connections to data source                        | 20                                                                        |
| LOADER_IGNORE_ERRORS  | If true, errors in download are skippeed with zero-value tiles | False                                                                     |
| NSPD_STORAGE_URL      | URL pattern to NSPD data storage with {image_id} placeholder   | `http://rasters-controller-api.service.consul:5150/v2/rasters/{image_id}` |
| NSPD_STORAGE_TIMEOUT  | Timeout on request to NSPD storage server, seconds             | 300                                                                       |
| X_ACTOR_ID            | actor ID to connect to internal NSPD/rk data sources           | ""                                                                        |

