from urban.functional.utils.angleutils import azimuth_from_vector, elevation_from_vector
from urban.functional.metaangles import check_meta_angles, compute_meta_angles
from gpdadapter import FeatureCollection
from urban.base.defaults import SAT_ELEVATION_TAG, SUN_ELEVATION_TAG, SAT_AZIMUTH_TAG, SUN_AZIMUTH_TAG
import numpy as np
from loguru import logger
import pytest
from typing import Final
import os
from tests.testutils.generate_test_files import generate_meta, generate_wl_markup, generate_sh_markup,\
    default_sat_azimuth, default_sat_elevation, default_sun_azimuth, default_sun_elevation


def test_azimuth_from_vector():
    assert azimuth_from_vector((1, 0)) == pytest.approx(90)
    assert azimuth_from_vector((100, 0)) == pytest.approx(90)
    assert azimuth_from_vector((0, 1)) == pytest.approx(0)
    assert azimuth_from_vector((0, -1)) == pytest.approx(-180)
    assert azimuth_from_vector((-1, 0)) == pytest.approx(-90)

    assert azimuth_from_vector((-2, 2)) == pytest.approx(-45)
    assert azimuth_from_vector((1, 1)) == pytest.approx(45)

    assert azimuth_from_vector((5, 7)) == pytest.approx(35.537677792)


def test_elevation_from_vector():
    assert elevation_from_vector((1, 0), 1) == pytest.approx(45)
    assert elevation_from_vector((3, -4), 5) == pytest.approx(45)
    assert elevation_from_vector((3, -4), 0) == pytest.approx(0)
    assert elevation_from_vector((3, -4), 0) == pytest.approx(0)


def test_compute_meta_angles():
    meta = generate_meta(with_angles=False)
    wll = generate_wl_markup()
    shl = generate_sh_markup()
    assert not check_meta_angles(meta)
    meta = compute_meta_angles(meta, wll, shl)
    assert check_meta_angles(meta)
    assert meta[0, SAT_AZIMUTH_TAG] == pytest.approx(default_sat_azimuth)
    assert meta[0, SAT_ELEVATION_TAG] == pytest.approx(default_sat_elevation)
    assert meta[0, SUN_AZIMUTH_TAG] == pytest.approx(default_sun_azimuth)
    assert meta[0, SUN_ELEVATION_TAG] == pytest.approx(default_sun_elevation)
