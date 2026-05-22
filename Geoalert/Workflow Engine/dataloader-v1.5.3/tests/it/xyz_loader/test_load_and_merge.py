import os
import pytest
import rasterio
import numpy as np
import maploader
from maploader.errors import TileNotReadable


REGION_1 = {
    "type": "Polygon",
    "coordinates": [
        [
            [
                41.71634316444397,
                36.04331291934319
            ],
            [
                41.71774059534073,
                36.04331291934319
            ],
            [
                41.71774059534073,
                36.04443849786634
            ],
            [
                41.71634316444397,
                36.04443849786634
            ],
            [
                41.71634316444397,
                36.04331291934319
            ]
        ]
    ]
}

URL = 'http://127.0.0.1:8000/tiles/{filetype}/{dtype}/{count}/{{z}}/{{x}}/{{y}}'

FILETYPES = ['png', 'jpg']
COUNTS = [1, 3, 4]


@pytest.mark.parametrize("count", COUNTS)
def test_download_png_default_params(count, test_data):
    """
    Loading with default paramtets must produce 3-channel data from anything in range from 1 to 4
    1- and 2- band data are interpreted as monochrome (with alpha in case of 2)
    3- and 4- band data is RGB (with optional alpha)

    Alpha channel, when present, must be transformed to a nodata (or bitmask?)
    """
    filetype = 'png'
    dtype = 'uint8'
    output_fp = os.path.join(test_data, "output_{filetype}_{count}_{dtype}.tif".format(filetype=filetype,
                                                                                       count=count,
                                                                                       dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    maploader.download(
        url,
        zoom=18,
        geometry=REGION_1,
        output_fp=output_fp)

    with rasterio.open(output_fp) as result:
        assert result.count == 3
        # single-channel image must be broadcasted to 3 channels
        if count == 1:
            assert np.mean(result.read(1)) == 10 and \
                   np.mean(result.read(2)) == 10 and \
                   np.mean(result.read(3)) == 10
        elif count in (3, 4):
            assert np.mean(result.read(1)) == 10 and \
                   np.mean(result.read(2)) == 20 and \
                   np.mean(result.read(3)) == 30


@pytest.mark.parametrize("count", [1, 3])
def test_download_jpg_default_params(count, test_data):
    """
    Loading with default paramtets must produce 3-channel data from anything in range from 1 to 4
    1- and 2- band data are interpreted as monochrome (with alpha in case of 2)
    3- and 4- band data is RGB (with optional alpha)

    Alpha channel, when present, must be transformed to a nodata (or bitmask?)
    """
    filetype = 'jpg'
    dtype = 'uint8'
    output_fp = os.path.join(test_data, "output_{filetype}_{count}_{dtype}.tif".format(filetype=filetype,
                                                                                       count=count,
                                                                                       dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    maploader.download(
        url,
        zoom=18,
        geometry=REGION_1,
        output_fp=output_fp)

    with rasterio.open(output_fp) as result:
        assert result.count == 3
        # single-channel image must be broadcasted to 3 channels
        if count == 1:
            assert np.mean(result.read(1)) == 10 and \
                   np.mean(result.read(2)) == 10 and \
                   np.mean(result.read(3)) == 10
        elif count == 3:
            assert np.mean(result.read(1)) == 10 and \
                   np.mean(result.read(2)) == 20 and \
                   np.mean(result.read(3)) == 30


def test_png_1_band(test_data):
    filetype = 'png'
    dtype = 'uint8'
    count = 1
    output_fp = os.path.join(test_data, "output_{filetype}_{count}_{dtype}.tif".format(filetype=filetype,
                                                                                       count=count,
                                                                                       dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    maploader.download(
        url,
        zoom=18,
        nchannels=1,
        geometry=REGION_1,
        output_fp=output_fp)

    with rasterio.open(output_fp) as result:
        assert result.count == count
        assert np.mean(result.read(1)) == 10


def test_png_3_bands(test_data):
    filetype = 'png'
    dtype = 'uint8'
    count = 3
    output_fp = os.path.join(test_data, "output_{filetype}_{count}_{dtype}.tif".format(filetype=filetype,
                                                                                       count=count,
                                                                                       dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    maploader.download(
        url,
        zoom=18,
        bands=3,
        nchannels=3,
        geometry=REGION_1,
        output_fp=output_fp)

    with rasterio.open(output_fp) as result:
        assert result.count == 3
        assert np.mean(result.read(1)) == 10 and \
               np.mean(result.read(2)) == 20 and \
               np.mean(result.read(3)) == 30


def test_png_4_bands(test_data):
    filetype = 'png'
    dtype = 'uint8'
    count = 4
    output_fp = os.path.join(test_data, "output_{filetype}_{count}_{dtype}.tif".format(filetype=filetype,
                                                                                       count=count,
                                                                                       dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    maploader.download(
        url,
        zoom=18,
        bands=1,
        nchannels=4,
        geometry=REGION_1,
        output_fp=output_fp)

    with rasterio.open(output_fp) as result:
        assert result.count == 4
        assert np.mean(result.read(1)) == 10 and \
               np.mean(result.read(2)) == 20 and \
               np.mean(result.read(3)) == 30 and \
               np.mean(result.read(4)) == 255


def test_wrong_bands_num(test_data):
    filetype = 'png'
    dtype = 'uint8'

    # ==== expected 1 bands, got 4 bands ==== #
    count = 4
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            nchannels=1,
            geometry=REGION_1,
            output_fp=output_fp)

    # ==== expected 1 bands, got 3 bands ==== #
    count = 3
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            nchannels=1,
            geometry=REGION_1,
            output_fp=output_fp)

    # ==== expected 1 bands, got 4 bands ==== #
    count = 4
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            nchannels=1,
            geometry=REGION_1,
            output_fp=output_fp)

    # ==== expected 3 bands, got 1 bands ==== #
    count = 1
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            bands=3,
            nchannels=3,
            geometry=REGION_1,
            output_fp=output_fp)

    # ==== expected 4 bands, got 3 bands ==== #
    count = 3
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            nchannels=4,
            geometry=REGION_1,
            output_fp=output_fp)

    # ==== expected 4 bands, got 1 bands ==== #

    count = 1
    output_fp = os.path.join(test_data, "nofile.tif".format(filetype=filetype, count=count, dtype=dtype))
    url = URL.format(filetype=filetype, count=count, dtype=dtype)
    with pytest.raises(TileNotReadable):
        maploader.download(
            url,
            zoom=18,
            nchannels=4,
            geometry=REGION_1,
            output_fp=output_fp)