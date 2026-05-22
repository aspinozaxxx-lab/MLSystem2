import os.path
import pytest
from pathlib import Path
from urban import (Compose, MeasureHeight, GenerateFootprints, GenerateRoofs, ComputeMetaAngles,
                   MeasureShift, CorrectShift, Brick)
from urban.functional import io
from typing import Final
from urban.base.defaults import SUN_ELEVATION_TAG, SAT_ELEVATION_TAG, SUN_AZIMUTH_TAG, SAT_AZIMUTH_TAG,\
    DEFINITIVE_HEIGHT_TAG


DATA_PATH: Final[str] = 'tests/test_data/synthetic/heights'
VEC_EXT: Final[str] = '.geojson'


@pytest.fixture
def output_files():
    output_roofs = 'output_roofs'
    output_footprints = 'output_footprints'

    yield output_roofs, output_footprints

    (Path(DATA_PATH)/(output_footprints + VEC_EXT)).unlink(missing_ok=True)
    (Path(DATA_PATH)/(output_roofs + VEC_EXT)).unlink(missing_ok=True)


def test_compute_meta_angles(output_files):
    out_meta, _ = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'shadows_labels'
    walls = 'walls_labels'

    pipeline = Compose(bricks=[ComputeMetaAngles(
        shadows=shadows,
        walls=walls,
        meta=meta,
        output=out_meta,
        )])

    pipeline(folder)
    meta_fc = io.read_fc(folder, out_meta)
    assert meta_fc.loc[0, SAT_ELEVATION_TAG] == pytest.approx(61.6229, abs=1.0)
    assert meta_fc.loc[0, SAT_AZIMUTH_TAG] == pytest.approx(88.40018, abs=1.0)
    assert meta_fc.loc[0, SUN_ELEVATION_TAG] == pytest.approx(47.879517, abs=1.0)
    assert meta_fc.loc[0, SUN_AZIMUTH_TAG] == pytest.approx(157.27017, abs=1.0)


def test_compute_meta_angles_ready(output_files):
    out_meta, _ = output_files
    folder = Path(DATA_PATH)
    meta = 'meta'
    shadows = 'shadows_labels'
    walls = 'walls_labels'

    pipeline = Compose(bricks=[ComputeMetaAngles(
        shadows=shadows,
        walls=walls,
        meta=meta,
        output=out_meta,
        )])

    pipeline(folder)
    meta_fc = io.read_fc(folder, out_meta)
    assert meta_fc.loc[0, SAT_ELEVATION_TAG] == pytest.approx(56.12, abs=1.0)
    assert meta_fc.loc[0, SAT_AZIMUTH_TAG] == pytest.approx(85.75, abs=1.0)
    assert meta_fc.loc[0, SUN_ELEVATION_TAG] == pytest.approx(41.37, abs=1.0)
    assert meta_fc.loc[0, SUN_AZIMUTH_TAG] == pytest.approx(158.66, abs=1.0)


def test_measure_height(output_files):
    output_roofs, _ = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'shadows'
    walls = 'walls'
    roofs = 'roofs'

    pipeline = Compose(bricks=[MeasureHeight(roofs=roofs,
                                      walls=walls,
                                      shadows=shadows,
                                      meta=meta,
                                      output=output_roofs)])
    pipeline(folder)
    result = io.read_fc(folder, output_roofs)
    # The building drawn, with the angles proposed, is taken from real image and is approximately 72 meters tall
    assert result[0, DEFINITIVE_HEIGHT_TAG] == pytest.approx(83.5, abs=1.5)


def test_measure_shift(output_files):
    output_roofs, _ = output_files
    folder = Path(DATA_PATH)

    walls = 'walls'
    roofs = 'roofs'
    x_shift_tag = '_x_shift'
    y_shift_tag = '_y_shift'

    pipeline = Compose(bricks=[MeasureShift(roofs=roofs,
                                     walls=walls,
                                     max_shift=50.0,
                                     rt_buffer=1.0,
                                     closing=0.0,
                                     simplify=0.0,
                                     max_iterations=30,
                                     x_shift_tag=x_shift_tag,
                                     y_shift_tag=y_shift_tag,
                                     output=output_roofs)])
    pipeline(folder)
    result = io.read_fc(folder, output_roofs)
    # The differential evolution method is stochastic in nature, there is an error in the result
    assert result[0, x_shift_tag] == pytest.approx(43.5, abs=1.5)
    assert result[0, y_shift_tag] == pytest.approx(2.1, abs=0.9)


def test_correct_shift(output_files):
    output_roofs, _ = output_files
    folder = Path(DATA_PATH)

    walls = 'wall_1'
    footprints = 'footprints_shifted_1'
    x_shift_tag = '_x_shift'
    y_shift_tag = '_y_shift'
    confidence_shift_tag = '_confidence_shift'
    corr_x_shift_tag = '_corr_x_shift'
    corr_y_shift_tag = '_corr_y_shift'
    corr_confidence_shift_tag = '_corr_confidence_shift'
    corr_threshold = 0.2

    pipeline = Compose(bricks = [CorrectShift(footprints=footprints,
                                     walls=walls,
                                     x_shift_tag=x_shift_tag,
                                     y_shift_tag=y_shift_tag,
                                     confidence_shift_tag=confidence_shift_tag,
                                     corr_x_shift_tag=corr_x_shift_tag,
                                     corr_y_shift_tag=corr_y_shift_tag,
                                     corr_confidence_shift_tag=corr_confidence_shift_tag,
                                     corr_threshold=corr_threshold,
                                     rt_buffer=1.0,
                                     closing=0.0,
                                     simplify=0.0,
                                     output=output_roofs)])
    pipeline(folder)
    result = io.read_fc(folder, output_roofs)
    assert result[0, corr_x_shift_tag] == pytest.approx(-19.4, abs=0.5)
    assert result[0, corr_y_shift_tag] == pytest.approx(-0.7, abs=0.5)


def test_measure_height_without_walls(output_files):

    output_roofs, _ = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'shadows'
    walls = 'empty_walls'
    roofs = 'roofs'

    pipeline = Compose(bricks=[MeasureHeight(roofs=roofs,
                                      walls=walls,
                                      shadows=shadows,
                                      meta=meta,
                                      output=output_roofs)])
    # must work without walls either
    pipeline(folder)


def test_measure_height_without_shadows(output_files):

    output_roofs, _ = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'empty_walls'
    walls = 'walls'
    roofs = 'roofs'

    pipeline = Compose(bricks=[MeasureHeight(roofs=roofs,
                                      walls=walls,
                                      shadows=shadows,
                                      meta=meta,
                                      output=output_roofs)])
    # must work without walls either
    pipeline(folder)
    result = io.read_fc(folder, output_roofs)
    # The building drawn, with the angles proposed, is taken from real image and is approximately 72 meters tall
    assert result.loc[0, DEFINITIVE_HEIGHT_TAG] == pytest.approx(83.5, abs=1.5)


def test_generate_footprints(output_files):
    output_roofs, output_footprints = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'shadows'
    walls = 'walls'
    roofs = 'roofs'
    footprints = 'footprints'

    pipeline = Compose(bricks=[MeasureHeight(roofs=roofs,
                                      walls=walls,
                                      shadows=shadows,
                                      meta=meta,
                                      output=output_roofs,
                                      height_range=(3, 200)),
                        GenerateFootprints(roofs=output_roofs,
                                           output=output_footprints)])
    pipeline(folder)
    result = io.read_fc(folder, output_footprints)[0, 'geometry']
    gt = io.read_fc(folder, footprints)[0, 'geometry']

    # The result must have good intersection with ground truth
    assert (result.intersection(gt)).area/(result.union(gt)).area > 0.7


def test_generate_roofs(output_files):
    output_roofs, output_footprints = output_files
    folder = Path(DATA_PATH)
    meta = 'meta_maxar'
    shadows = 'shadows'
    walls = 'walls'
    roofs = 'roofs'
    footprints = 'footprints'

    pipeline = Compose(bricks=[MeasureHeight(roofs=roofs,
                                      walls=walls,
                                      shadows=shadows,
                                      meta=meta,
                                      output=output_roofs,
                                      height_range=(3, 200)),
                        GenerateFootprints(roofs=output_roofs,
                                           output=output_footprints),
                        GenerateRoofs(footprints=output_footprints,
                                      output=output_roofs)])
    pipeline(folder)
    result = io.read_fc(folder, output_roofs)[0, 'geometry']
    gt = io.read_fc(folder, roofs)[0, 'geometry']

    # The result must have good intersection with ground truth
    assert (result.intersection(gt)).area/(result.union(gt)).area > 0.7
