"""Объектный экспорт сохраняет фильтрацию формы и защиту обрезанных участков."""

import sys
import json
import zipfile
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import rasterio
import yaml
from pydantic import BaseModel
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape

from mlsystem2.inference.contracts import ObjectSeparationConfig
from mlsystem2.models.contracts import ModelSpec
from mlsystem2.training_ui_api import _model_export
from mlsystem2.training_ui_api._geoalert_vector import register_vector_bricks
from mlsystem2.training_ui_api._model_export import build_geoalert_pipeline_yaml


def _pipeline(profile, config):
    separation = ObjectSeparationConfig(profile=profile, boundary_threshold=0.65)
    return yaml.safe_load(build_geoalert_pipeline_yaml(
        model_name="zu500_orto", sample_size=512, input_channels=3,
        probability_output=True, object_output=True,
        object_separation=separation.model_dump(), postprocess_config=config,
    ))["config"]["bricks"]


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
@pytest.mark.parametrize("mode,filter_name", [
    ("keep_compact", "FilterNonCompactObjects"),
    ("remove_compact", "FilterCompactObjects"),
])
def test_object_export_preserves_boundary_options_and_filter_order(profile, mode, filter_name):
    segmentation, vectorize, processing = _pipeline(profile, {
        "postprocess.min_area_m2": 10,
        "postprocess.min_hole_area_m2": 2,
        "postprocess.filter_compact_objects.enabled": True,
        "postprocess.filter_compact_objects.mode": mode,
        "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.3,
        "postprocess.filter_compact_objects.max_bbox_ratio": 4,
        "postprocess.smooth.enabled": True,
        "postprocess.simplify_m": 0.5,
    })
    assert segmentation["_class"] == "ObjectF1Segmentation"
    assert segmentation["separation"]["profile"] == profile
    assert segmentation["separation"]["boundary_threshold"] == 0.65
    assert segmentation["adapter"]["output_dtype"] == "float32"
    assert vectorize == {
        "_class": "VectorizeMasks", "input_rasters": ["object_instances"],
        "output_fcs": ["output"], "value_property_name": "instance_id",
        "mark_boundary_objects": True, "boundary_tolerance_pixels": 1.0,
        "boundary_tag": "_touches_raster_boundary",
        **({"match_instance_edges": True} if profile == "object_f1_v2" else {}),
    }
    assert processing["input"] == processing["output"] == "output"
    filters = processing["bricks"]
    assert [brick["_class"] for brick in filters] == [
        "FilterSmallObjects", "RemoveSmallHoles", filter_name, "Smooth",
        "CoverageSimplify" if profile == "object_f1_v2" else "Simplify", "RemoveTags",
    ]
    for brick in (filters[0], filters[2]):
        assert brick["preserve_boundary_objects"] is True
        assert brick["boundary_tag"] == vectorize["boundary_tag"]
    assert filters[2]["min_isoperimetric_quotient"] == 0.3
    assert filters[2]["max_bbox_ratio"] == 4
    assert filters[-2] == {
        "_class": "CoverageSimplify" if profile == "object_f1_v2" else "Simplify", "rate": 0.5,
    }
    assert filters[-1]["tags"] == [vectorize["boundary_tag"]]


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
@pytest.mark.parametrize("config", [None, {}, {
    "postprocess.filter_compact_objects.enabled": False,
    "postprocess.filter_compact_objects.mode": "keep_compact",
}, {"postprocess.min_area_m2": 10}])
def test_boundary_protection_not_enabled_without_compact_filter(profile, config):
    bricks = _pipeline(profile, config)
    assert bricks[1] == {
        "_class": "VectorizeMasks", "input_rasters": ["object_instances"],
        "output_fcs": ["output"], "value_property_name": "instance_id",
        **({"match_instance_edges": True} if profile == "object_f1_v2" else {}),
    }
    if config and config.get("postprocess.min_area_m2"):
        assert bricks[2]["bricks"] == [
            {"_class": "FilterSmallObjects", "min_area": 10.0, "area_tag": "area"},
        ]
    else:
        assert len(bricks) == 2


class _Feature(dict):
    @property
    def geometry(self):
        return self["geometry"]


class _Collection(list):
    @classmethod
    def from_features(cls, features, crs):
        collection = cls(_Feature(geometry=shape(feature["geometry"]), **feature["properties"])
                         for feature in features)
        collection.crs = crs
        return collection

    @property
    def empty(self):
        return not self

    @property
    def geometry(self):
        return [feature.geometry for feature in self]

    @geometry.setter
    def geometry(self, geometries):
        for feature, geometry in zip(self, geometries, strict=True):
            feature["geometry"] = geometry

    def filter(self, predicate):
        return _Collection(feature for feature in self if predicate(feature))

    def apply(self, function):
        return [function(feature) for feature in self]

    def drop(self, column, *, axis, inplace):
        assert axis == 1 and inplace
        for feature in self:
            del feature[column]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return [feature[key[1]] for feature in self]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if np.isscalar(value):
                value = [value] * len(self)
            for feature, item in zip(self, value, strict=True):
                feature[key[1]] = item
        else:
            super().__setitem__(key, value)


@pytest.fixture
def vector_runtime(monkeypatch):
    """Штатные расширения с минимальной заменой внешнего окружения Geoalert."""
    registry = {}
    collections = {}

    class PolygonProcessingBrick(BaseModel):
        input: str = "output"
        output: str | None = None

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            registry[cls.__name__] = cls

    class VectorizeMasks(PolygonProcessingBrick):
        input_rasters: list[str]
        output_fcs: list[str]
        value_property_name: str

        def __call__(self, path):
            for raster, vector in zip(self.input_rasters, self.output_fcs, strict=True):
                with rasterio.open(path / f"{raster}.tif") as source:
                    instances = source.read(1)
                    collections[vector] = _Collection(
                        _Feature(geometry=shape(geometry), **{self.value_property_name: int(value)})
                        for geometry, value in shapes(instances, mask=instances > 0, transform=source.transform)
                    )

    class FilterSmallObjects(PolygonProcessingBrick):
        pass

    urban = ModuleType("urban")
    urban.VectorizeMasks = VectorizeMasks
    urban.FilterSmallObjects = FilterSmallObjects
    base = ModuleType("urban.base.brick")
    base.Brick = BaseModel
    base.PolygonProcessingBrick = PolygonProcessingBrick
    functional = ModuleType("urban.functional")
    io = ModuleType("urban.functional.io")
    io.read_fc = lambda path, name, crs: collections[name]
    io.save_fc = lambda collection, path, name: collections.update({name: collection})
    functional.io = io
    gpdadapter = ModuleType("gpdadapter")
    gpdadapter.FeatureCollection = _Collection
    monkeypatch.setitem(sys.modules, "urban", urban)
    monkeypatch.setitem(sys.modules, "urban.base.brick", base)
    monkeypatch.setitem(sys.modules, "urban.functional", functional)
    monkeypatch.setitem(sys.modules, "gpdadapter", gpdadapter)
    register_vector_bricks()
    return registry, collections


@pytest.mark.parametrize("profile", ["object_f1", "object_f1_v2"])
@pytest.mark.parametrize("mode,expected_ids", [
    ("keep_compact", {1, 2, 4}), ("remove_compact", {3, 4}),
])
def test_generated_pipeline_filters_instances_independently_and_keeps_edge(
    tmp_path, vector_runtime, profile, mode, expected_ids,
):
    instances = np.zeros((40, 50), dtype=np.int32)
    instances[10:20, 5:15] = 1
    instances[10:20, 15:25] = 2  # Соседние компактные участки имеют общий край.
    instances[25:27, 5:35] = 3  # Узкая внутренняя полоса должна отфильтроваться.
    instances[0:2, 5:35] = 4  # Такой же фрагмент на краю сохраняется.
    with rasterio.open(tmp_path / "object_instances.tif", "w", driver="GTiff",
                       width=50, height=40, count=1, dtype="int32", crs="EPSG:3857",
                       nodata=0, transform=from_origin(100, 200, 1, 1)) as target:
        target.write(instances, 1)
    _, vectorize, processing = _pipeline(profile, {
        "postprocess.filter_compact_objects.enabled": True,
        "postprocess.filter_compact_objects.mode": mode,
    })
    registry, collections = vector_runtime
    registry[vectorize["_class"]](**{key: value for key, value in vectorize.items() if key != "_class"})(tmp_path)
    features = collections["output"]
    assert {feature["instance_id"] for feature in features} == {1, 2, 3, 4}
    assert {feature["instance_id"] for feature in features if feature[vectorize["boundary_tag"]]} == {4}
    filter_config, remove_tags = processing["bricks"]
    compact_filter = registry[filter_config["_class"]](
        input="output", **{key: value for key, value in filter_config.items() if key != "_class"},
    )
    result = compact_filter.process(features)
    assert {feature["instance_id"] for feature in result} == expected_ids
    assert remove_tags == {"_class": "RemoveTags", "tags": [vectorize["boundary_tag"]]}
    assert all(feature.geometry.equals(shape(geometry))
               for feature in result
               for geometry, value in shapes(instances, mask=instances > 0, transform=from_origin(100, 200, 1, 1))
               if int(value) == feature["instance_id"])


@pytest.mark.parametrize("profile", ["legacy", "next_gen2", "object_f1", "object_f1_v2"])
def test_export_metadata_distinguishes_current_algorithm_from_checkpoint(profile, monkeypatch):
    separation = ObjectSeparationConfig(profile="object_f1_v2", boundary_threshold=0.65).model_dump()
    metadata = {"pipeline_variant": profile, "sample_size": 512,
                "inference_context": 0, "confidence_threshold": 0.5}
    loaded = SimpleNamespace(
        artifact=SimpleNamespace(metadata=metadata),
        model=SimpleNamespace(model=object(), spec=ModelSpec(
            name="smp_segformer_b0", input_channels=3, output_channels=1,
            parameters={"pipeline_variant": profile, "object_separation": separation},
        )),
    )
    monkeypatch.setattr(_model_export, "_load_binary_checkpoint", lambda _: loaded)
    monkeypatch.setattr(_model_export, "_checkpoint_task_schema", lambda _: ("binary", []))

    def export(**kwargs):
        kwargs["onnx_path"].write_bytes(b"test-onnx")

    monkeypatch.setattr(_model_export, "_export_segmentation_mask_onnx", export)
    monkeypatch.setattr(_model_export, "_export_binary_mask_onnx", export)
    archive = _model_export.build_triton_model_export_zip(
        model_name="zu500_orto", checkpoint_filename="best.pt", checkpoint_bytes=b"checkpoint", sample_size=None,
    )
    try:
        with zipfile.ZipFile(archive.zip_path) as source:
            current = json.loads(source.read("export_metadata.json"))
        assert current["checkpoint_metadata"] == metadata
        assert "object_separation_algorithm" not in metadata
        if profile == "object_f1_v2":
            assert current["inference_profile"] == "object_f1_instances_v2_interior_geodesic_fp32_shared_edges_v1"
            assert current["object_separation_algorithm"] == "interior_geodesic_v1"
            assert current["object_simplification_algorithm"] == "shared_edges_v1"
            assert current["inference_precision"] == "fp32_no_tf32"
            assert current["cuda_use_tf32"] is False
            assert current["object_separation"] == separation
        else:
            if profile == "next_gen2":
                assert current["inference_profile"] == "notebook_valid_data_full_coverage_v3"
            else:
                assert "inference_profile" not in current
            assert "object_separation_algorithm" not in current
            assert "object_simplification_algorithm" not in current
            assert "inference_precision" not in current
            assert "cuda_use_tf32" not in current
    finally:
        archive.cleanup()
