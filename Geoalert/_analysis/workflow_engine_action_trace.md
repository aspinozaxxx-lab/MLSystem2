# Workflow Engine action trace

## Workflow Engine endpoints
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/api/DefinitionController.java:32-34` - REST controller `/api/v0/definitions`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/api/DefinitionController.java:75-82` - `POST /api/v0/definitions` читает YAML и вызывает `workflowDefinitionImporter.importWorkflowDefinition(yml)`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/api/WorkflowController.java:33-35` - REST controller `/api/v0/workflows`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/api/WorkflowController.java:86-99` - `POST /api/v0/workflows` создает workflow через `workflowService.create(workflow)`.

## Action registry
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/entity/definition/action/Action.java:25-62` - enum `Action` задает action names и worker names.
- `Action.java:54-57` - `INFERENCE("inference", Optional.of("inference"), false)`; значит action `inference` маршрутизируется в worker `inference`.
- `Action.java:26-62` - другие worker names: `dataloader`, `source-validator`, `raster-processor`, `vector-processor`.

## Workflow definition import
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/service/definition/WorkflowDefinitionImporter.java:88-104` - stage definition получает `Action.fromActionName(stage.action)` и сохраняет stage params.
- `WorkflowDefinitionImporter.java:107-115` - `dependsOn` разрешается в связи stage dependencies.

## Stage start and queue routing
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/service/action/stagestarter/TaskStageStarter.java:18-23` - stage starter выбирает `TaskCreator` по `Action` и создает tasks.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/action/inference/InferenceTaskCreator.java:43-52` - для каждого `RAW_RASTER` artifact создается inference task.
- `InferenceTaskCreator.java:62-90` - stage params фильтруются; если есть `model`, параметр `pipeline.<model>` перекладывается в `pipeline`; без `pipeline` stage падает.
- `InferenceTaskCreator.java:102-144` - `TaskMessage.input` получает params, AOI, blocks и `source_data` с `input.tif`.
- `InferenceTaskCreator.java:146-167` - `TaskMessage.output.output_data` получает `output.geojson` и/или `output.tar`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/service/queue/TaskMessage.java:8-12` - message содержит `task_id`, `processing_id`, `input`, `output`, `runcheck_url`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/service/queue/MessageSender.java:65-74` - routing key берется из `action.getWorkerName()` и сообщение отправляется в `task.exchange`.
- `Geoalert/Workflow Engine/workflow-engine-v1.1.1/src/main/java/ru/skoltech/aeronetlab/urban/service/queue/QueueConfig.java:97-115` - очереди создаются по worker names и биндинги используют routing key без суффикса queue.

## Inference worker
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/main.py:6-12` - worker создает `InferenceMessageHandlder` и слушает очередь через `QueueClient`.
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/message.py:46-50` - вход worker: `aoi`, `pipeline`, `source_data`, `blocks`.
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/message_handler.py:22-31` - handler передает `pipeline`, artifacts, AOI и blocks в `DataProcessor.run`.
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/data_processor.py:37-45` - base64 pipeline сохраняется в `pipeline.yaml` и загружается через `urban.Compose.load`.
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/data_processor.py:59-64` - artifacts скачиваются в workdir, затем выполняется `pipeline(self.workdir)`.

## Blockers
- `Geoalert/Workflow Engine/inference-v1.5.5/inference/message_handler.py:14-17` и `Dockerfile:14-16` содержат санитизированные секреты/токены, поэтому полноценный запуск worker без восстановления конфигурации заблокирован.
- Docker/WSL на машине отсутствуют, локальная сборка Docker route не выполнялась.
