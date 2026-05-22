import rasterio
from shapely.geometry import shape
import pytest
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from inference.utils import mask_local_raster_by_aoi, aoi_fraction_cover
from rasterio.crs import CRS

crs = CRS.from_epsg(3857)


@pytest.fixture
def create_test_files():

    # L-shape covering a top and right parts of the raster with 50 meters (100 pixels) stripes.
    aoi = shape({"type": "Polygon", "coordinates": [[[10000, 20000],
                                                     [10500, 20000],
                                                     [10500, 19500],
                                                     [10450, 19500],
                                                     [10450, 19950],
                                                     [10000, 19950],
                                                     [10000, 20000]]]})
    aoi = shape(aoi)
    #aoi_latlon = shape(rasterio.warp.transform_geom(src_crs="EPSG:3857",
    #                                               dst_crs="EPSG:4326",
    #                                               geom=aoi))

    tempdir = TemporaryDirectory()
    workdir = Path(tempdir.name)
    filename = "input.tif"
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
    with rasterio.open(workdir / filename, 'w', **profile) as dst:
        dst.write(raster)

    yield workdir, filename, aoi

    tempdir.cleanup()


def test_aoi_fraction_cover(create_test_files):
    workdir, filename, aoi = create_test_files
    with rasterio.open(workdir/filename) as src:
        assert aoi_fraction_cover(src=src, aoi=aoi, aoi_crs=crs) == pytest.approx(19/100)


def test_mask_file_without_nodata(create_test_files):
    workdir, filename, aoi = create_test_files
    with rasterio.open(workdir/filename) as src:
        assert np.all(src.read_masks(1) == 255)
        data = src.read()
        assert np.all(data == 1)
    mask_local_raster_by_aoi(workdir/filename, aoi, aoi_crs=crs)

    with rasterio.open(workdir/filename) as src:
        mask = src.read_masks(1)
        data = src.read()
        # probe three points in mask
        assert mask[0, 0] == 255
        assert mask[0, 9] == 255
        assert mask[1, 1] == 0
        assert mask[9, 0] == 0
        # count masked pixels
        assert np.count_nonzero(mask) == 19
        # image should not be altered
        assert np.all(data==1)

