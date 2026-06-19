# Инструкция по созданию псевдоразметки через Geoalert

Этот документ описывает, как Codex должен делать псевдоразметку снимков через Geoalert по лучшей обученной сети MLSystem2. Инструкция рассчитана на ручную оркестрацию действиями Codex: проверить модель, экспортировать checkpoint в Triton, запустить Geoalert `Compose`, собрать GeoJSON, проверить результат и скопировать его пользователю.

Псевдоразметка через Training UI не использует этот путь: UI запускает `mlsystem2.training_ui_api._pseudo_runner`, загружает `checkpoints/best.pt` напрямую через PyTorch (`inference_backend=pytorch_one_off`) и не создает Triton model archive, pipeline YAML или запись в model repository. Этот Geoalert/Triton runbook нужен только для явного ручного production-инференса или экспорта модели по отдельному запросу.

Не добавляй новый CLI и не создавай постоянный модуль в `src`, если пользователь просит только псевдоразметку. Временные скрипты для одноразового серверного прогона можно создавать в runtime-папке запуска или подавать в Python через stdin. Код MLSystem2 менять не нужно, если нет явной ошибки.

## 1. Основные пути

Рабочий сервер:

```bash
gpu-mlserver
```

Репозиторий MLSystem2 на сервере:

```bash
/opt/mlsystem2/repo
```

Geoalert inference:

```bash
/opt/geoalert/inference
/opt/geoalert/inference/.venv
```

Triton:

```bash
container: geoalert-triton
model repository: /opt/geoalert/triton_models
pipelines: /opt/geoalert/pipelines
server runs: /opt/geoalert/runs
http endpoint: http://127.0.0.1:8000
```

Данные:

```bash
MLMarkup: /data/MLMarkup
prepared images: /data/mlsystem2/prepared_images
```

Не используй `/data/MLSystem2`. Не создавай и не используй symlink `/data/mlmarkup`.

Локальная папка, куда пользователь обычно просит складывать псевдоразметку:

```text
D:\Projects\razmetka\
```

## 2. Как выбрать сеть

Если пользователь просит "лучшую сеть", выбирай checkpoint по лучшей validation-метрике из последней релевантной HPO-сессии или MLflow experiment. Для binary segmentation основной ориентир:

```text
val/best_threshold_pixel_f1
```

Если есть HPO-отчет, сначала смотри:

```bash
/opt/hpo/report/<имя_сессии>/session_state.json
/opt/hpo/report/<имя_сессии>/best_trials.md
/opt/hpo/report/<имя_сессии>/trials.jsonl
```

Если HPO-отчет недоступен, смотри MLflow:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.get_experiment_by_name("<experiment_name>")
runs = mlflow.search_runs(
    [exp.experiment_id],
    order_by=["metrics.train/best_threshold_pixel_f1 DESC"],
    max_results=20,
)
print(runs[["run_id", "status", "metrics.train/best_threshold_pixel_f1"]])
```

Для псевдоразметки используй `best.pt`, а не `final.pt`, если `best.pt` существует. Threshold бери из checkpoint metadata `val_best_threshold`. Если его нет, используй лучший threshold из MLflow history. Если нет и его, используй `train.threshold` из конфига, но явно напиши это в отчете.

Текущий важный пример последней HPO-сессии по `ВырубкиТест`:

```text
HPO session: /opt/hpo/report/deforestation_test2_smp_segformer_b2_0206
MLflow experiment: Deforestation_Test_2
champion trial: 0003
run_id: 356d5fdb2a244e76a5d6863b34300d0d
model: smp_segformer_b2
checkpoint: /opt/mlsystem2/runtime/hpo/deforestation_test2_smp_segformer_b2_0206/scratch/trial_0003/checkpoints/best.pt
best F1: 0.6938652209164091
best epoch: 46
threshold: 0.8
```

## 3. Как выбрать датасет

Если пользователь говорит "датасет вырубок", без слова "тест", используй полный датасет:

```bash
/data/MLMarkup/Вырубки/deforestation.txt
/data/MLMarkup/Вырубки/deforestation.geojson
```

Если пользователь говорит `ВырубкиТест`, используй тестовый датасет:

```bash
/data/MLMarkup/ВырубкиТест/deforestation.txt
/data/MLMarkup/ВырубкиТест/deforestation.geojson
```

Снимки ищи в:

```bash
/data/mlsystem2/prepared_images
```

Сцены брать из `*.txt`, а не из GeoJSON. Для каждой сцены ищи TIFF по `Path.stem`; учитывай варианты с суффиксом `_cog` и без него.

Проверка сцен и снимков:

```bash
cd /opt/mlsystem2/repo
.venv/bin/python - <<'PY'
from pathlib import Path

scenes_file = Path("/data/MLMarkup/Вырубки/deforestation.txt")
images_dir = Path("/data/mlsystem2/prepared_images")

scenes = [line.strip() for line in scenes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
files = list(images_dir.rglob("*.tif")) + list(images_dir.rglob("*.tiff"))
index = {path.stem: path for path in files}

missing = []
for scene in scenes:
    base = scene[:-4] if scene.endswith("_cog") else scene
    found = None
    for candidate in (scene, base, f"{base}_cog"):
        if candidate in index:
            found = index[candidate]
            break
    if found is None:
        missing.append(scene)
    else:
        print(scene, "->", found)

print("scene_count", len(scenes))
print("missing", missing)
PY
```

## 4. Preflight перед запуском

Перед псевдоразметкой проверь, что сервер свободен от обучения, Triton жив, а нужные пути есть:

```bash
ssh gpu-mlserver
ps -eo pid,pgid,stat,etime,cmd | awk '/python/ && /mlsystem2.cli.train/ && !/awk/ {print}'
nvidia-smi
test -d /opt/mlsystem2/repo
test -d /opt/geoalert/inference
test -d /opt/geoalert/triton_models
test -d /data/MLMarkup
test -d /data/mlsystem2/prepared_images
docker ps --filter name=geoalert-triton
```

Если активен HPO/train и пользователь попросил закончить HPO, останови только текущий train process group:

```bash
kill -TERM -<PGID>
```

Не убивай unrelated процессы.

## 5. Экспорт checkpoint в Triton

Экспорт делай через публичный API модуля `models`: `mlsystem2.models.api.load_checkpoint`. Не создавай свою фабрику моделей, если существующий checkpoint содержит `model_spec`.

Создавай уникальное имя Triton-модели. Не перезаписывай существующую модель `mlsystem2_deforestation`, если пользователь явно не попросил. Хороший формат имени:

```text
mlsystem2_<class_or_dataset>_<session_or_model>_<trial>_thr<threshold>
```

Пример для champion последней HPO-сессии:

```text
mlsystem2_deforestation_test2_hpo0003_thr080
```

Экспортный wrapper для binary segmentation должен возвращать `uint8` mask:

```text
sigmoid(logits) > threshold
```

Вход Geoalert/Triton остается raw `float32` `[1,4,H,W]`. Не добавляй внешнюю нормализацию. Если модель MLSystem2 требует scaling, он уже находится внутри модели или был учтен при обучении.

Удобнее подавать длинный Python-скрипт на сервер через stdin, чтобы не ломаться на кавычках PowerShell:

```powershell
@'
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone

import torch

from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest

checkpoint = Path("/opt/mlsystem2/runtime/hpo/deforestation_test2_smp_segformer_b2_0206/scratch/trial_0003/checkpoints/best.pt")
model_name = "mlsystem2_deforestation_test2_hpo0003_thr080"
model_root = Path("/opt/geoalert/triton_models") / model_name
version_dir = model_root / "1"
onnx_path = version_dir / "model.onnx"
threshold = 0.8

if model_root.exists():
    shutil.rmtree(model_root)
version_dir.mkdir(parents=True, exist_ok=True)

loaded = load_checkpoint(LoadCheckpointRequest(checkpoint_uri=str(checkpoint), map_location="cpu"))
model = loaded.model.model.eval()

class BinaryMaskWrapper(torch.nn.Module):
    def __init__(self, model, threshold: float) -> None:
        super().__init__()
        self.model = model
        self.threshold = float(threshold)

    def forward(self, x):
        logits = self.model(x.float())
        if hasattr(logits, "logits"):
            logits = logits.logits
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return (torch.sigmoid(logits) > self.threshold).to(torch.uint8)

wrapper = BinaryMaskWrapper(model, threshold).eval()
dummy = torch.zeros((1, 4, 1024, 1024), dtype=torch.float32)
with torch.no_grad():
    output = wrapper(dummy)
print("dry_output", tuple(output.shape), output.dtype)

torch.onnx.export(
    wrapper,
    dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["mask"],
    opset_version=17,
    dynamic_axes={"input": {2: "height", 3: "width"}, "mask": {0: "batch", 2: "height", 3: "width"}},
    do_constant_folding=True,
)

config = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 4, -1, -1 ]
  }}
]
output [
  {{
    name: "mask"
    data_type: TYPE_UINT8
    dims: [ 1, 1, -1, -1 ]
  }}
]
instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
'''
(model_root / "config.pbtxt").write_text(config, encoding="utf-8")

metadata = {
    "exported_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint": str(checkpoint),
    "checkpoint_metadata": loaded.artifact.metadata,
    "model_spec": loaded.model.spec.model_dump(mode="json"),
    "threshold": threshold,
    "source_run_id": "356d5fdb2a244e76a5d6863b34300d0d",
    "source_trial": "0003",
    "hpo_experiment": "Deforestation_Test_2",
    "onnx_path": str(onnx_path),
    "onnx_size_bytes": onnx_path.stat().st_size,
    "output": "uint8 mask after sigmoid(logits) > threshold",
}
(model_root / "export_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metadata, ensure_ascii=False, indent=2))
'@ | ssh gpu-mlserver "cd /opt/mlsystem2/repo && .venv/bin/python -"
```

После экспорта проверь ONNX:

```bash
cd /opt/mlsystem2/repo
.venv/bin/python - <<'PY'
import onnx

path = "/opt/geoalert/triton_models/mlsystem2_deforestation_test2_hpo0003_thr080/1/model.onnx"
model = onnx.load(path)
print("inputs", [(item.name, [d.dim_value or d.dim_param for d in item.type.tensor_type.shape.dim]) for item in model.graph.input])
print("outputs", [(item.name, [d.dim_value or d.dim_param for d in item.type.tensor_type.shape.dim]) for item in model.graph.output])
print("initializers", len(model.graph.initializer), "nodes", len(model.graph.node))
onnx.checker.check_model(model)
print("onnx_check_ok")
PY
```

Важно: `config.pbtxt` должен совпадать с формой ONNX output. Если ONNX output `[1,1,H,W]`, указывай:

```text
dims: [ 1, 1, -1, -1 ]
```

Если ONNX output `[batch,1,H,W]`, можно использовать:

```text
dims: [ -1, 1, -1, -1 ]
```

При несовпадении Triton не загрузит модель.

## 6. Загрузка модели в Triton

Triton должен работать в explicit model control mode, чтобы модели из `/opt/geoalert/triton_models` не загружались в GPU-память автоматически. Для восстановления контейнера с тем же image и read-only model repository используй:

```bash
docker stop geoalert-triton
docker rm geoalert-triton
docker run -d \
  --name geoalert-triton \
  --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /opt/geoalert/triton_models:/models:ro \
  nvcr.io/nvidia/tritonserver:25.03-py3 \
  tritonserver \
  --model-repository=/models \
  --strict-model-config=true \
  --log-verbose=0 \
  --model-control-mode=explicit
```

Не добавляй `--load-model`, если нет отдельного решения держать конкретную production-модель в памяти. После старта проверь, что сервер жив и repository index не содержит загруженных `READY` моделей:

```bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do
  if curl -sf http://127.0.0.1:8000/v2/health/ready >/dev/null; then
    echo ready_after_${i}s
    break
  fi
  sleep 1
done

curl -s -X POST http://127.0.0.1:8000/v2/repository/index
docker logs geoalert-triton --since 2m 2>&1 | tail -120
nvidia-smi
```

Для ручного production-прогона загружай только выбранную модель и после завершения выгружай ее:

```bash
curl -s -X POST http://127.0.0.1:8000/v2/repository/models/mlsystem2_deforestation_test2_hpo0003_thr080/load
curl -s http://127.0.0.1:8000/v2/models/mlsystem2_deforestation_test2_hpo0003_thr080
curl -s http://127.0.0.1:8000/v2/models/mlsystem2_deforestation_test2_hpo0003_thr080/ready
curl -s -X POST http://127.0.0.1:8000/v2/repository/models/mlsystem2_deforestation_test2_hpo0003_thr080/unload
```

Во время Geoalert `Compose` модель должна быть `READY`; после unload она не должна оставаться загруженной в GPU-памяти.

## 7. Geoalert pipeline YAML

Создай отдельный pipeline YAML под конкретную Triton-модель:

```bash
cat > /opt/geoalert/pipelines/mlsystem2_deforestation_test2_hpo0003_thr080_triton.yaml <<'YAML'
version: 0.1.4
config:
  _class: Compose
  inputs:
    - input.tif
  outputs:
    - output.geojson
  bricks:
    - _class: SplitRaster
      input: input
      input_ext: tif
      output:
        - RED
        - GRN
        - BLU
        - NIR
    - _class: Segmentation
      bounds: 0
      sample_size:
        - 1024
        - 1024
      input_rasters:
        - RED
        - GRN
        - BLU
        - NIR
      output_labels:
        - mask
      nodata: 0
      adapter:
        _class: TritonAdapter
        name: mlsystem2_deforestation_test2_hpo0003_thr080
        host: 127.0.0.1
        port: 8000
        protocol: http
        input_dtype: float32
        input_ndim: 4
        output_ndim: 3
        output_dtype: uint8
        timeout: 120
        n_retries: 1
    - _class: VectorizeMasks
      input_rasters:
        - mask
      output_fcs:
        - output
YAML
```

`sample_size` обычно `1024 x 1024`. Это не обязано совпадать с train tile size: Geoalert режет большой снимок на окна для инференса.

## 8. Запуск Geoalert по датасету

Для одноразового запуска используй прямой Python-скрипт в `/opt/geoalert/inference/.venv`. Он должен:

1. прочитать `scenes_file`;
2. найти TIFF в `/data/mlsystem2/prepared_images`;
3. для каждой сцены создать workdir с symlink `input.tif`;
4. вызвать `Compose`;
5. скопировать `output.geojson` в per-scene output;
6. добавить свойства источника в каждый feature;
7. объединить все features в итоговый GeoJSON;
8. записать `report.json` с количеством объектов, ошибками и статистикой маски.

Для полного датасета вырубок задай:

```text
SCENES_FILE = /data/MLMarkup/Вырубки/deforestation.txt
RUN_ROOT = /opt/geoalert/runs/pseudo_deforestation_hpo0003_thr080_<дата>
```

Для `ВырубкиТест` задай:

```text
SCENES_FILE = /data/MLMarkup/ВырубкиТест/deforestation.txt
RUN_ROOT = /opt/geoalert/runs/pseudo_deforestation_test2_hpo0003_thr080_<дата>
```

Итоговый файл называй понятно:

```text
pseudo_labels_deforestation_hpo0003_thr080.geojson
pseudo_labels_deforestation_test2_hpo0003_thr080.geojson
```

Минимальный алгоритм запуска:

```python
import json
import shutil
import sys
import time
from pathlib import Path

PIPELINE = Path("/opt/geoalert/pipelines/mlsystem2_deforestation_test2_hpo0003_thr080_triton.yaml")
SCENES_FILE = Path("/data/MLMarkup/ВырубкиТест/deforestation.txt")
IMAGES_DIR = Path("/data/mlsystem2/prepared_images")
RUN_ROOT = Path("/opt/geoalert/runs/pseudo_deforestation_test2_hpo0003_thr080_20260602")
MERGED = RUN_ROOT / "pseudo_labels_deforestation_test2_hpo0003_thr080.geojson"
REPORT = RUN_ROOT / "report.json"

sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, "/opt/geoalert/inference")
sys.path.insert(0, "/opt/geoalert/inference/shims")
sys.path.insert(0, "/opt/geoalert/inference/modules/urban")
sys.path.insert(0, "/opt/geoalert/inference/modules/aeronet_raster")
sys.path.insert(0, "/opt/geoalert/inference/modules/gpdadapter")

from modules.urban.urban.base import compose, parser

def write_fc(path: Path, features: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")

def find_image(scene: str, index: dict[str, Path]) -> Path | None:
    base = scene[:-4] if scene.endswith("_cog") else scene
    for candidate in (scene, base, f"{base}_cog"):
        if candidate in index:
            return index[candidate]
    return None

started = time.time()
if RUN_ROOT.exists():
    shutil.rmtree(RUN_ROOT)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

scenes = [line.strip() for line in SCENES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
files = list(IMAGES_DIR.rglob("*.tif")) + list(IMAGES_DIR.rglob("*.tiff"))
index = {path.stem: path for path in files}

pipeline_config = parser.parse_config(str(PIPELINE))
pipeline = compose.Compose.from_config(pipeline_config["config"])

all_features = []
scene_reports = []
failures = []
missing = []

for number, scene in enumerate(scenes, 1):
    image = find_image(scene, index)
    if image is None:
        missing.append(scene)
        scene_reports.append({"scene_id": scene, "status": "missing_image", "feature_count": 0})
        continue

    workdir = RUN_ROOT / "work" / scene
    outdir = RUN_ROOT / "per_scene" / scene
    if workdir.exists():
        shutil.rmtree(workdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (workdir / "input.tif").symlink_to(image)

    item_started = time.time()
    try:
        pipeline(str(workdir))
        output = workdir / "output.geojson"
        data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {"type": "FeatureCollection", "features": []}
        features = data.get("features") or []
        for feature in features:
            props = feature.setdefault("properties", {})
            props["scene_id"] = scene
            props["class_slug"] = "deforestation"
            props["class_name"] = "Вырубки"
            props["source_model"] = "smp_segformer_b2"
            props["source_hpo_trial"] = "0003"
            props["source_run_id"] = "356d5fdb2a244e76a5d6863b34300d0d"
            props["source_checkpoint"] = "/opt/mlsystem2/runtime/hpo/deforestation_test2_smp_segformer_b2_0206/scratch/trial_0003/checkpoints/best.pt"
            props["source_threshold"] = 0.8
            props["triton_model"] = "mlsystem2_deforestation_test2_hpo0003_thr080"
        write_fc(outdir / "deforestation.geojson", features)
        all_features.extend(features)
        scene_reports.append({
            "scene_id": scene,
            "status": "ok",
            "image": str(image),
            "feature_count": len(features),
            "elapsed_sec": round(time.time() - item_started, 3),
        })
    except Exception as exc:
        failures.append({"scene_id": scene, "image": str(image), "error": repr(exc)})
        scene_reports.append({"scene_id": scene, "status": "failed", "image": str(image), "feature_count": 0, "error": repr(exc)})
        write_fc(outdir / "deforestation.geojson", [])

write_fc(MERGED, all_features)
summary = {
    "status": "ok" if not failures and not missing else "partial",
    "scenes_file": str(SCENES_FILE),
    "scene_count": len(scenes),
    "processed": sum(1 for item in scene_reports if item["status"] == "ok"),
    "failed": len(failures),
    "missing_images": len(missing),
    "feature_count": len(all_features),
    "output_geojson": str(MERGED),
    "elapsed_sec": round(time.time() - started, 3),
    "source": {
        "trial": "0003",
        "run_id": "356d5fdb2a244e76a5d6863b34300d0d",
        "threshold": 0.8,
    },
    "scenes": scene_reports,
    "failures": failures,
    "missing": missing,
}
REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

Для одной сцены можно использовать тот же алгоритм, просто `scenes_file` будет содержать один scene id.

## 9. Проверка результата

После Geoalert обязательно проверь:

- процесс завершился без failures;
- итоговый GeoJSON читается как `FeatureCollection`;
- число features совпадает с отчетом;
- если есть `mask.tif`, он не пустой, если ожидается положительная разметка;
- у features проставлены `scene_id`, `class_slug`, `source_checkpoint`, `source_threshold`, `triton_model`;
- на сервере не осталось train-процессов.

Проверка локального или серверного GeoJSON:

```python
import json
from pathlib import Path

path = Path("D:/Projects/razmetka/pseudo_labels_deforestation_test2_hpo0003_thr080.geojson")
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("type"))
print(len(data.get("features") or []))
print((data.get("features") or [{}])[0].get("properties", {}))
```

Если `feature_count=0`, это не всегда ошибка. Проверь `mask.tif`: если mask тоже нулевая, возможно threshold слишком высокий или модель реально ничего не нашла. Если это неожиданно, попробуй threshold из validation sweep ниже, но не меняй threshold молча: создай отдельную Triton-модель и отдельный output filename с новым threshold.

## 10. Копирование пользователю

Копируй итоговый GeoJSON и отчет в локальную папку:

```powershell
New-Item -ItemType Directory -Force -Path 'D:\Projects\razmetka' | Out-Null
scp gpu-mlserver:<server_run_root>/<output>.geojson 'D:\Projects\razmetka\<output>.geojson'
scp gpu-mlserver:<server_run_root>/report.json 'D:\Projects\razmetka\<output>_report.json'
```

Имена файлов должны содержать датасет, источник модели или trial и threshold:

```text
D:\Projects\razmetka\pseudo_labels_deforestation_hpo0003_thr080.geojson
D:\Projects\razmetka\pseudo_labels_deforestation_hpo0003_thr080_report.json
D:\Projects\razmetka\pseudo_labels_deforestation_test2_hpo0003_thr080.geojson
D:\Projects\razmetka\pseudo_labels_deforestation_test2_hpo0003_thr080_report.json
```

## 11. Что написать пользователю в конце

В финальном ответе укажи:

- какая сеть использована: model, HPO session, trial, run id;
- checkpoint;
- threshold;
- какие снимки обработаны;
- число объектов в GeoJSON;
- была ли маска пустой или нет;
- локальные пути к GeoJSON и report;
- если были failures или missing images, перечисли их.

Пример:

```text
Псевдоразметка готова. Использована лучшая сеть HPO: smp_segformer_b2, trial 0003, run_id 356d5fdb2a244e76a5d6863b34300d0d, threshold 0.8. Обработан датасет ВырубкиТест, 1 снимок, получено 125 объектов. Файл лежит в D:\Projects\razmetka\pseudo_labels_deforestation_test2_hpo0003_thr080.geojson, отчет - рядом.
```

## 12. Готовый запрос для Codex

Пользователь может дать такой запрос:

```text
Сделай псевдоразметку по docs/inference_instruction.md через Geoalert, используя лучший smp_segformer_b2 из последней HPO-сессии по вырубкам. Обработай снимки датасета Вырубки и положи итоговый GeoJSON и отчет в D:\Projects\razmetka.
```

Если нужно обработать тестовый датасет:

```text
Сделай псевдоразметку по docs/inference_instruction.md через Geoalert, используя лучший smp_segformer_b2 из последней HPO-сессии Deforestation_Test_2. Обработай датасет ВырубкиТест и положи итоговый GeoJSON и отчет в D:\Projects\razmetka.
```
