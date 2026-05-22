# Vector Importer

## Env variables

### Database config
| Name              | Description                                            | Default value   |
|-------------------|--------------------------------------------------------|-----------------|
| DATABASE_URI      | database hostname                                      | vector-database |
| DATABASE_NAME     | the name of database will be used                      | vector_db       |
| DATABASE_USER     | database user                                          | postgres        |
| DATABASE_PASSWORD | database password                                      | 1234Qq          |
| DATABASE_SCHEMA   | database schema                                        | public          |
| FLYWAY_TABLE          | migration table name                                   | schema_history  |

### Minio config
| Name             | Description                                                      | Default value |
|------------------|------------------------------------------------------------------|---------------|
| minio.host       | Minio host                                                       | None          |
| minio.port       | Minio port                                                       | None          |
| minio.access.key | Minio access key ID                                              | None          |
| minio.secret.key | Minio access secret                                              | None          |

### Queue config
| Name                  | Description                                | Default value |
|-----------------------|--------------------------------------------|---------------|
| rabbitmq.host         | RabbitMQ host                              | queue         |
| rabbitmq.node.port    | RabbitMQ port                              | 5672          |
| rabbitmq.default.user | RabbitMQ username                          | user          |
| rabbitmq.default.pass | RabbitMQ password                          | password      |
| rabbitmq.max.priority | RabbitMQ maximum task priority             | 10            |
| input.queue           | Queue name postfix. Do not change          | .tasks.queue  |
| output.queue          | Queue name postfix. Do not change          | .result.queue |

### App config
| Name           | Description                                            | Default value    |
|----------------|--------------------------------------------------------|------------------|
| worker.name    | WE Worker name. Do not change                          | vector-processor |
| LOGGING_LEVEL  | app logging level                                      | INFO             |
| SHOW_SQL       | sets up logging of SQL requests for debugging purposes | false            |
| SQL_LOG_LEVEL  | logging level of sql statements                        | INFO             |

