from pathlib import Path
import rasterio
import pytest
import os
import shutil
import numpy as np

from urban import Compose, AddConstant, ApplyMask
from .generate_files import create_tiff_file


@pytest.fixture(scope='module')
def get_child_parent_files():
    folder = './tests/test_data/tmp1/'
    child_name = 'child'
    parent_name = 'parent'
    profile = {'height': 1500,
               'width': 2000,
               'count': 1,
               'dtype': 'uint8',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (10.0, 0, 100000, 0, -10.0, 20000)}

    os.makedirs(folder, exist_ok=True)
    create_tiff_file(filename=os.path.join(folder, child_name + '.tif'), value=100, matrix='ones', **profile)
    create_tiff_file(filename=os.path.join(folder, parent_name + '.tif'), matrix='eye', **profile)

    yield folder, child_name, parent_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def test_add_constant(get_file):
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output1']
    folder = Path(folder)
    constant = 20.0

    pipeline = Compose(bricks=[AddConstant(input_rasters=input_rasters,
                                           output_rasters=output_labels,
                                           constant=constant)])
    pipeline(folder)

    with rasterio.open(folder / (output_labels[0] + '.tif')) as output, rasterio.open(
            folder / (name + '.tif')) as input:
        assert ((input.read() + constant) == output.read()).all()


def test_apply_mask_dtype(get_child_parent_files):
    folder, child_name, parent_name = get_child_parent_files
    output_name = 'output'
    folder = Path(folder)
    reverse_parent = False

    pipeline = Compose(bricks=[ApplyMask(parent_mask=parent_name,
                                         child_masks=[child_name],
                                         out_masks=[output_name],
                                         reverse_parent=reverse_parent,
                                         mask_operation='mask')])
    pipeline(folder)

    with (rasterio.open(folder / (output_name + '.tif')) as src1,
          rasterio.open(folder / (child_name + '.tif')) as src2):
        assert src1.profile['dtype'] == src2.profile['dtype']


def test_apply_mask_reverse(get_child_parent_files):
    folder, child_name, parent_name = get_child_parent_files
    output_name = 'output'
    folder = Path(folder)
    reverse_parent = True

    pipeline = Compose(bricks=[ApplyMask(parent_mask=parent_name,
                                         child_masks=[child_name],
                                         out_masks=[output_name],
                                         reverse_parent=reverse_parent,
                                         mask_operation='mask')])
    pipeline(folder)

    with rasterio.open(folder / (output_name + '.tif')) as src1:
        out_arr = src1.read()

    with rasterio.open(folder / (parent_name + '.tif')) as src2:
        parent_arr = src2.read()

    with rasterio.open(folder / (child_name + '.tif')) as src3:
        child_arr = src3.read()

    if reverse_parent:
        parent_arr = (parent_arr == 0).astype(np.uint8)

    expected_out = np.where(parent_arr != 0, child_arr, 0).astype(child_arr.dtype)

    assert (expected_out == out_arr).all()
