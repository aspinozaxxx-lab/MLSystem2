## Development and testing
Create your personal gitlab token with API scope to manage access for queue_client library installation

Unit tests:
```bash
make GITLAB_TOKEN= utest
```
Integration tests:
```bash
make GITLAB_TOKEN= itest
```

### Descriptions

| Pipeline name                |  Classes  |                                 Models                                 | Description                                                                                                                 | Serving  |
| ---------------------------- | :-------: | :--------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------- | :------: |
| buildings_instance_v001.yaml | buildings |             russia_roofs_instance<br>russia_classification             | Buildings instance segmentation with classification and postprocessing (simplification, aligning)                           | 0.5, 0.6 |
| buildings_instance_v002.yaml | buildings | russia_roofs_instance<br>russia_shadows_walls<br>russia_classification | Buildings instance segmentation with classification and postprocessing (simplification, aligning) and **height estimation** | 0.5, 0.6 |
| forest_semantic_v001.yaml    |  forest   |                                 forest                                 | Forest semantic segmentation **without** any postprocessing                                                                 | 0.5, 0.6 |
| forest_semantic_v002.yaml    |  forest   |                                 forest                                 | Forest semantic segmentation **with** height regression                                                                     |   0.6    |
| roads_semantic_v001.yaml     |   roads   |                                 roads                                  | Roads semantic segmentation **without** any postprocessing                                                                  | 0.5, 0.6 |

### Task Example (queue message)

Exmaple #1

```json
{
  "task_id": "1",
  "input": {
    "pipeline": "buildings_instance_v001.yaml",
    "source_data": [
      {
        "name": "input.tif",
        "path": "s3://data/wf-001/input.tif"
      }
    ]
  },
  "output": {
    "bucket": "data",
    "filename": "wf-001/output.geojson"
  }
}
```

Exmaple #2

```json
{
  "task_id": "2",
  "input": {
    "pipeline": "buildings_instance_v002.yaml",
    "source_data": [
      {
        "name": "input.tif",
        "path": "s3://data/wf-001/input.tif"
      },
      {
        "name": "shadows_labels.geojson",
        "path": "s3://data/wf-001/shadows_labels.geojson"
      },
      {
        "name": "walls_labels.geojson",
        "path": "s3://data/wf-001/walls_labels.geojson"
      },
      {
        "name": "meta.geojson",
        "path": "s3://data/wf-001/meta.geojson"
      }
    ]
  },
  "output": {
    "bucket": "data",
    "filename": "wf-001/output.geojson"
  }
}
```

### Response (queue message)

```json
{
  "task_id": 1,
  "status": 0,
  "error_message": null
}
```

Status:

- 0 - SUCCESS
- 1 - FAIL

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
| Name           | Description                                  | Default value       |
|----------------|----------------------------------------------|---------------------|
| WORKER_NAME    | WE Worker name. Do not change                | inference           |
| LOG_LEVEL      | Logging level                                | INFO                |
| VSI_CACHE_SIZE | GDAL setting                                 | 1000000000          |
| OUTPUT_BUCKET  | Additional bucket for arbitrary output files | inference-artifacts |

