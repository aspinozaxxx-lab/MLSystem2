See https://geoalert.fibery.io/Documentation/Data-requirements-API-description-691

After cloning the repository run:

```
git submodule update --init --recursive
```


# To run and test

## unit tests on data-validator-lib
```commandline
./test.sh
```

## manual e2e tests
read the output. The logs of source-validator will show if it is all OK

```commandline
./test.sh manual
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
| Name           | Description                   | Default value    |
|----------------|-------------------------------|------------------|
| WORKER_NAME    | WE Worker name. Do not change | source-validator |
| LOG_LEVEL      | Logging level                 | INFO             |


