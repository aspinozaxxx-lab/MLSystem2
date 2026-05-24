# Geoalert inference deploy

Дата проверки: 2026-05-24 16:40 MSK.

## Где развернуто

- Source Geoalert inference: `/opt/mlsystem2/repo/Geoalert/Workflow Engine/inference-v1.5.5`
- Рабочая копия: `/opt/geoalert/inference`
- Python venv: `/opt/geoalert/inference/.venv`
- Local pipeline YAML: `/opt/geoalert/pipelines/mlsystem2_deforestation_triton.yaml`
- Smoke workdir: `/opt/geoalert/smoke/kanopus_1024`
- Debug tensors: `/opt/geoalert/smoke/debug_tensors`

`/data/MLSystem2` не использовался. Symlink `/data/mlmarkup` не создавался.

## Python dependencies

Локальные Geoalert-модули установлены editable:

- `aeronet_raster 0.2.6` -> `/opt/geoalert/inference/modules/aeronet_raster`
- `gpdadapter 1.0.0` -> `/opt/geoalert/inference/modules/gpdadapter`
- `urban 0.1.4` -> `/opt/geoalert/inference/modules/urban`

Ключевые runtime packages в venv:

- `numpy==1.26.4`, `pydantic==2.13.4`, `loguru==0.4.1`
- `rasterio==1.4.4`, `geopandas==1.1.3`, `fiona==1.10.1`, `shapely==2.1.2`, `pyproj==3.7.2`
- `opencv-python-headless==4.11.0.86`, `scipy==1.17.1`, `scikit-image==0.26.0`, `scikit-learn==1.8.0`
- `rasterstats==0.21.0`, `topojson==1.10`, `sknw==0.15`, `tritonclient==2.68.0`

Compatibility notes:

- Exact old pins from Geoalert requirements are not Python 3.12-compatible (`pyproj==3.2.1`, `PyYAML==5.4.1`, `scipy==1.5.0`), so compatible current wheels were installed without changing brick logic.
- `we-queue-client[minio]==1.5.2` was not available from the server-visible package indexes. Queue worker was not started. For local smoke, a minimal shim `/opt/geoalert/inference/shims/we_queue_client/utils.py` provides only `log_time` used by `Compose`/SAM imports.
- `osgeo.gdal` is taken from system package `python3-gdal` through `PYTHONPATH=/usr/lib/python3/dist-packages`.
- Working copy has a Shapely 2 compatibility import shim: `cascaded_union` fallback to `unary_union` in `.../shapely_ext/_algorithms/voronoi.py`.

## Triton

Docker/NVIDIA check passed:

- Docker: `29.4.1`
- GPU: `NVIDIA GeForce RTX 5090`, driver `580.159.03`
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`: passed

Triton container:

```bash
docker run -d --name geoalert-triton --gpus all --restart unless-stopped \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /opt/geoalert/triton_models:/models:ro \
  nvcr.io/nvidia/tritonserver:25.03-py3 \
  tritonserver --model-repository=/models --strict-model-config=true --log-verbose=0
```

Status:

- container: `geoalert-triton`, running, restart policy `unless-stopped`
- HTTP ready: `http://127.0.0.1:8000/v2/health/ready`
- model endpoint: `http://127.0.0.1:8000/v2/models/mlsystem2_deforestation`
- model status: `READY`
- stats after smoke: `inference_count=4`, `fail.count=0`

## Triton model repo

Model repository: `/opt/geoalert/triton_models`

Files:

- `/opt/geoalert/triton_models/mlsystem2_deforestation/1/model.onnx` (`109921239` bytes)
- `/opt/geoalert/triton_models/mlsystem2_deforestation/config.pbtxt`
- `/opt/geoalert/triton_models/mlsystem2_deforestation/export_metadata.json`

`config.pbtxt`:

```text
name: "mlsystem2_deforestation"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 1, 4, -1, -1 ]
  }
]
output [
  {
    name: "mask"
    data_type: TYPE_UINT8
    dims: [ -1, -1, -1, -1 ]
  }
]
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

## Checkpoint/model used

- Checkpoint: `/opt/mlsystem2/runtime/first_train/scratch/checkpoints/best.pt`
- Source code for MLSystem2 model loader/export: `/opt/mlsystem2/repo`
- Model: `segformer_b2`, `input_channels=4`, `output_channels=1`, `pretrained=false`
- Checkpoint metadata: `label=best`, `epoch=1`, `val_pixel_f1=0.018755369439880563`, `val_loss=0.9891750366588676`
- Train tile size used for Geoalert smoke: `1024`
- Export: ONNX opset 17, dynamic H/W, input `N,C,H,W`
- Wrapper output: `sigmoid(logits) > 0.75`, cast to `uint8` mask

`onnx==1.21.0` was installed into `/opt/mlsystem2/repo/.venv` for export/checking.

## Pipeline YAML

```yaml
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
        name: mlsystem2_deforestation
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
```

## Input image

Prepared Kanopus source:

`/data/mlsystem2/prepared_images/kanopus/Olskij/KV4_29142_28015-00_KANOPUS_20230505_001746_14.L2.PMS.SCN03.tif`

For fast smoke, `input.tif` is a 1024 x 1024 crop from source window `x=1024, y=1024`:

```bash
gdal_translate -quiet -srcwin 1024 1024 1024 1024 \
  /data/mlsystem2/prepared_images/kanopus/Olskij/KV4_29142_28015-00_KANOPUS_20230505_001746_14.L2.PMS.SCN03.tif \
  /opt/geoalert/smoke/kanopus_1024/input.tif
```

Input metadata: 4 bands, `uint8`, EPSG:3857, resolution `2.560542892464513`.

## Smoke command

```bash
cd /opt/geoalert/inference
source .venv/bin/activate
export PYTHONPATH=/opt/geoalert/inference/shims:/opt/geoalert/inference/modules:/opt/geoalert/inference/modules/urban:/opt/geoalert/inference/modules/aeronet_raster:/opt/geoalert/inference/modules/gpdadapter:/usr/lib/python3/dist-packages:$PYTHONPATH
export GEOALERT_DEBUG_TENSORS=1
export GEOALERT_DEBUG_TENSORS_DIR=/opt/geoalert/smoke/debug_tensors
python - <<'PY'
from urban.base import Compose
pipeline = Compose.load('/opt/geoalert/pipelines/mlsystem2_deforestation_triton.yaml')
outputs = pipeline('/opt/geoalert/smoke/kanopus_1024')
print(outputs)
PY
```

Result: passed.

## Output artifacts

Smoke workdir: `/opt/geoalert/smoke/kanopus_1024`

- `RED.tif`, `GRN.tif`, `BLU.tif`, `NIR.tif`
- `mask.tif` (`uint8`, 1024 x 1024, unique values `[0]`, sum `0`)
- `output.geojson` (`FeatureCollection`, `features=0`)

Debug tensor sample confirms Geoalert -> Triton ABI:

- input tensor: shape `[1, 4, 1024, 1024]`, dtype `float32`, value range `175..255`
- output tensor: shape `[1, 1, 1024, 1024]`, dtype `uint8`, unique `[0]`

Empty mask/output is acceptable for this smoke because the checkpoint is an early epoch-1 B2 checkpoint and quality was not the pass criterion.

## Errors/blockers

Resolved:

- Direct `ssh root@31.192.104.147` failed without explicit key; SSH alias/key `gpu-mlserver` works.
- Private `we-queue-client[minio]==1.5.2` unavailable; direct `Compose` smoke used instead of worker queue.
- Python 3.12 incompatible old requirement pins; installed compatible wheels.
- Shapely 2 removed `cascaded_union`; working-copy compatibility import added.
- `osgeo.gdal` absent from isolated venv; system `python3-gdal` added via `PYTHONPATH`.
- `fiona` was missing for empty GeoJSON writing; installed `fiona==1.10.1`.
- Initial Triton config over-specified output shape `[1,1,-1,-1]`; ONNX metadata reports output dims as fully dynamic, so config now uses `[-1,-1,-1,-1]` while actual response remains `[1,1,H,W]`.

Open:

- Queue worker path is not validated because `we-queue-client` is unavailable.
- Full-scene inference was not run; smoke used a 1024 x 1024 crop to validate the inference path on GPU.
