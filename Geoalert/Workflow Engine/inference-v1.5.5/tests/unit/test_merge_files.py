import rasterio
import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from inference.utils import merge_files
from rasterio.crs import CRS

crs = CRS.from_epsg(3857)


@pytest.fixture
def create_test_files():
    # First raster
    tempdir = TemporaryDirectory()
    workdir = Path(tempdir.name)
    filename1 = workdir /"part_1.tif"
    size = 10
    profile = {'width': size,
               'height': size,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(50, 0, 10000, 0, -50, 20000),
               'crs': 'EPSG:3857',
               'nodata': 0}
    raster = np.ones(dtype='uint8',
                     shape=(3, size, size))
    with rasterio.open(filename1, 'w', **profile) as dst:
        dst.write(raster)

    # Second raster - shifted 450 meters left and 450 meters up
    filename2 = workdir /"part_2.tif"
    size = 10
    profile = {'width': size,
               'height': size,
               'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(50, 0, 9550, 0, -50, 20450),
               'crs': 'EPSG:3857',
               'nodata': 0}
    raster = np.ones(dtype='uint8',
                     shape=(3, size, size))*2
    with rasterio.open(filename2, 'w', **profile) as dst:
        dst.write(raster)

    yield workdir, filename1, filename2

    tempdir.cleanup()


def test_merge(create_test_files):
    workdir, filename1, filename2 = create_test_files
    merge_files([filename1, filename2], workdir/'output.tif')
    with rasterio.open(workdir/'output.tif') as src:
        # 2 images of 10*10 with 1-pixel overlap
        assert src.width == 19
        assert src.height == 19
        assert src.transform == pytest.approx(rasterio.Affine(50, 0, 9550, 0, -50, 20450))
        data = src.read(1)
        # data from second image
        assert data[0, 0] == 2
        assert data[9, 0] == 2
        assert data[0, 9] == 2
        # pixel on intersection is from SECOND due to "start from first" default strategy of VRT build
        assert data[9, 9] == 2
        # pixels from first image
        assert data[9, 10] == 1
        assert data[10, 9] == 1
        assert data[18, 18] == 1
        # pixels out of images
        assert data[0, 18] == 0
        assert data[18, 0] == 0
