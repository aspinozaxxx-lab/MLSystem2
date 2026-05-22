# Маршрут Mapflow -> Workflow Engine

## Entry points
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/rest/ProcessingResource.scala:86-90` - REST `POST /processings` читает `CreateAndRunProcessingJson` и вызывает `runProcessingService.createAndRun(input)(user)`.
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/graphql/schema/GraphQLSchema.scala:425-430` - GraphQL mutation `runProcessing` запускает уже созданный processing.

## Workflow request construction
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/service/WorkflowEngineService.scala:179-238` - `startWorkflow` читает workflow, AOI, processing, vector/raster layer, параметры processing и block config, затем формирует `RunWorkflowSummary`.
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/service/WorkflowEngineService.scala:260-265` - при `RequiredAction.start` вызывает `workflowEngine.postWorkflow(summary)`.
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/service/we/model/PostWorkflowRequest.scala:21-45` - из `RunWorkflowSummary` собирается payload для Workflow Engine.

## Workflow Engine client
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/DefaultConfig.scala:134` - URL берется из `WORKFLOW_ENGINE_URL`, дефолт `http://localhost:8060`.
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/service/we/ProductionWorkflowEngine.scala:68-78` - workflow definition отправляется multipart POST на `/api/v0/definitions`.
- `Geoalert/Mapflow/mapflow-api-v1.3.2/src/main/scala/io/geoalert/mapflow/service/we/ProductionWorkflowEngine.scala:87-93` - workflow запускается POST на `/api/v0/workflows`.

## Payload fields
- `areasOfInterest`: список из одного AOI, формируется из `summary.wf.geometry.geom` в `PostWorkflowRequest.scala:34-36`.
- `workflowDefinitionId`: `summary.wf.workflowDef.weId`, строка `PostWorkflowRequest.scala:36`.
- `system`: внешний system id, строка `PostWorkflowRequest.scala:37`.
- `processingId`: строковый UUID processing, строка `PostWorkflowRequest.scala:38`.
- `params`: параметры processing плюс `priority`, `vector-layer-id`, `raster-layer-uri`, строки `PostWorkflowRequest.scala:39-43`.
- `blocks`: список `{name, enabled}`, строка `PostWorkflowRequest.scala:44`.
- Если в params есть `url` с `s3://` или `source_type=tif`, Mapflow заменяет `source_type` на `local`: `PostWorkflowRequest.scala:22-32`.

## Confirmed / not confirmed
- Подтверждено кодом: Mapflow API не выполняет inference сам; он отправляет workflow definition и workflow payload в Workflow Engine.
- Подтверждено кодом: `inference` stage доходит до Workflow Engine как action name из workflow definition.
- Не подтверждено запуском: локальный HTTP route не проверен, потому что Docker/WSL отсутствуют, а Mapflow API содержит санитизированный Scala-код, который ломает сборку.
