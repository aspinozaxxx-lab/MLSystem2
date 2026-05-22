import sys; sys.path.append('/home/user3/work/code/urban/')  # ???

from urban.functional.postprocessing.flatten_multipolygons import merge_connected_polygons
from urban.functional import io
from gpdadapter import FeatureCollection
from pathlib import Path
import pytest


TEST_FILE_PATH = './tests/test_data/synthetic/merge_connected_polygons/'
TEST_FILE_NAME = 'intersected_and_not_intesected_polygons'


def test_zero_polygons():
    """
    Takes zero polygons return zero polygons
    """
    fc = FeatureCollection()
    result = merge_connected_polygons(fc)

    assert not result


def test_one_polygon_results_count():
    """
    Takes one polygon return one polygon
    """
    fc = io.read_fc(TEST_FILE_PATH, TEST_FILE_NAME)[0]
    result = merge_connected_polygons(fc)

    assert len(result) == 1


def test_many_polygons_results_count():
    """
    Takes 12 polygons in the file merge them to 5 polygons
    """
    fc = io.read_fc(TEST_FILE_PATH, TEST_FILE_NAME)
    result = merge_connected_polygons(fc)

    assert len(result) == 5
