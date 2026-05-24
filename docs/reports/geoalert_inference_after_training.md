# Отчет: Geoalert inference после B2 aug3 smart training

Дата: 2026-05-24.

## Контур

- MLSystem2 commit: `2dbe0c7873d51adc31852cfe50b5ba3f8e6aaf52`.
- Training run id: `36ebd154de454f06b7e8d5e728e34ff6`.
- Training run name: `deforestation_2405_6`.
- Checkpoint: `/opt/mlsystem2/runtime/first_train/b2_aug3_smart_30ep_scratch/checkpoints/best.pt`.
- Checkpoint epoch: `14`.
- Checkpoint val F1: `0.05610510493431182`.
- Geoalert inference working copy: `/opt/geoalert/inference`.
- Pipeline: `/opt/geoalert/pipelines/mlsystem2_deforestation_triton.yaml`.
- Output root: `/opt/geoalert/runs/deforestation_after_training`.
- Prepared images: `/data/mlsystem2/prepared_images/`.
- Scenes: `/data/MLMarkup/Вырубки/deforestation.txt`.

`/data/MLSystem2` не использовался. Symlink `/data/mlmarkup` не создавался.

## ONNX/Triton

Best checkpoint экспортирован в Triton repository:

- ONNX: `/opt/geoalert/triton_models/mlsystem2_deforestation/1/model.onnx`.
- ONNX size: `112660349` bytes.
- Opset: `18`.
- Input shape: `N,C,H,W`, dynamic H/W.
- Runtime sample size в Geoalert pipeline: `1024`.
- Threshold: `0.75`.
- Output: `uint8 mask after sigmoid + threshold`.
- Triton model: `mlsystem2_deforestation`.
- Triton status: `READY`.

`config.pbtxt` после экспорта:

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
    dims: [ 1, 1, -1, -1 ]
  }
]
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

## Summary

- Status: `ok`.
- Scene count: `35`.
- Processed: `35`.
- Failed: `0`.
- Missing inputs: `0`.
- Elapsed: `336.01 sec`.
- Total mask sum: `353604222`.
- Total output.geojson features: `18669`.

Сцены с нулевой маской:

- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN03`.
- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN02`.
- `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN01`.

## Per-scene results

`mask unique` записан как `value:pixel_count`.

| Scene | Input | Mask unique | Mask sum | Features | Status |
| --- | --- | --- | ---: | ---: | --- |
| `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN03.tif` | `0:240821697; 1:452574` | `452574` | `113` | `ok` |
| `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN02.tif` | `0:234680474; 1:546322` | `546322` | `85` | `ok` |
| `KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV4_18254_14029-01_KANOPUS_20210518_080116_20.L2.PMS.SCN01.tif` | `0:211034346; 1:109054` | `109054` | `48` | `ok` |
| `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN03.tif` | `0:212779498; 1:2033945` | `2033945` | `261` | `ok` |
| `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN02.tif` | `0:261404091; 1:2516976` | `2516976` | `330` | `ok` |
| `KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_13775_10997-01_KANOPUS_20210621_075153_83.L2.PMS.SCN01.tif` | `0:231964183; 1:603417` | `603417` | `200` | `ok` |
| `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN03.tif` | `0:225595189; 1:5978621` | `5978621` | `302` | `ok` |
| `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN02.tif` | `0:221919992; 1:5573638` | `5573638` | `421` | `ok` |
| `KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_24775_25670-02_KANOPUS_20230615_075909_7.L2.PMS.SCN01.tif` | `0:189488827; 1:8686057` | `8686057` | `417` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN07_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN07.tif` | `0:193333071; 1:1718812` | `1718812` | `364` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN06_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN06.tif` | `0:140901439; 1:2789313` | `2789313` | `390` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN05_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN05.tif` | `0:136789908; 1:6780552` | `6780552` | `549` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN04_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN04.tif` | `0:137407194; 1:6031258` | `6031258` | `539` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN03_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN03.tif` | `0:136774640; 1:6579760` | `6579760` | `700` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN02_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN02.tif` | `0:138792833; 1:4441411` | `4441411` | `542` | `ok` |
| `KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN01_cog` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV5_24818_25736-01_KANOPUS_20230618_035304_20.L2.PMS.SCN01.tif` | `0:142557550; 1:917481` | `917481` | `132` | `ok` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN06` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN06.tif` | `0:93343659; 1:49117896` | `49117896` | `1613` | `ok` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN05` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN05.tif` | `0:98084709; 1:44172801` | `44172801` | `1806` | `ok` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN04` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN04.tif` | `0:94570804; 1:47566964` | `47566964` | `2258` | `ok` |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN03.tif` | `0:104046991; 1:37983240` | `37983240` | `3042` | `ok` |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07.tif` | `0:138325660; 1:3816284` | `3816284` | `361` | `ok` |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN04` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN04.tif` | `0:128757094; 1:12990470` | `12990470` | `948` | `ok` |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03.tif` | `0:135434158; 1:6718106` | `6718106` | `592` | `ok` |
| `KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN02.tif` | `0:59312483; 1:8443` | `8443` | `10` | `ok` |
| `KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV6_25931_26639-01_KANOPUS_20230830_075057_8.L2.PMS.SCN01.tif` | `0:141520250; 1:5848` | `5848` | `14` | `ok` |
| `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN03.tif` | `0:125521247; 1:17482348` | `17482348` | `719` | `ok` |
| `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN02.tif` | `0:176430630; 1:41904526` | `41904526` | `968` | `ok` |
| `KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV6_29676_31877-02_KANOPUS_20240502_075932_8.L2.PMS.SCN01.tif` | `0:149416200; 1:35212367` | `35212367` | `470` | `ok` |
| `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN03.tif` | `0:232059435` | `0` | `0` | `ok` |
| `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN02.tif` | `0:224604170` | `0` | `0` | `ok` |
| `KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/wave_2_Upload_01/KV5_31621_35243-01_KANOPUS_20240907_075653_8.L2.PMS.SCN01.tif` | `0:195378243` | `0` | `0` | `ok` |
| `KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN04` | `/data/mlsystem2/prepared_images/kanopus/Hilokskij/KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN04.tif` | `0:218350337; 1:502843` | `502843` | `262` | `ok` |
| `KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN03` | `/data/mlsystem2/prepared_images/kanopus/Hilokskij/KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN03.tif` | `0:217582556; 1:75184` | `75184` | `62` | `ok` |
| `KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN02` | `/data/mlsystem2/prepared_images/kanopus/Hilokskij/KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN02.tif` | `0:216392472; 1:57528` | `57528` | `47` | `ok` |
| `KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN01` | `/data/mlsystem2/prepared_images/kanopus/Hilokskij/KV6_35415_39922-00_KANOPUS_20250513_033040_82.L2.PMS.SCN01.tif` | `0:193059039; 1:230183` | `230183` | `104` | `ok` |

## Вывод

Geoalert direct Compose path с Triton-моделью `mlsystem2_deforestation` валиден для полного списка сцен вырубок: все 35 сцен обработаны, входы найдены в `/data/mlsystem2/prepared_images/`, `output.geojson` получен для каждой сцены. Три сцены дали пустую маску и ноль features; остальные дали положительные пиксели и векторные объекты.
