import pytest
import rasterio
import os
import shutil
from pathlib import Path
from urban import Compose, Segmentation
from urban.bricks.adapters import MockAdapter
from .generate_files import create_tiff_file

BASIC_MODEL_PATH = 'tests/mock_models/basic_model.py'


def _run_segmentation_with_mock_adapter(input_rasters,
                                        output_labels,
                                        folder,
                                        res,
                                        crs,
                                        res_tolerance=0.01,
                                        max_upsampling=4,
                                        input_dtype='uint16',
                                        output_dtype='uint16',
                                        min_res=None,
                                        max_res=None):
    # model_kwargs include, most important, res and crs
    brick = Segmentation(
        adapter=MockAdapter(name='mock', path=BASIC_MODEL_PATH, input_dtype=input_dtype, output_dtype=output_dtype),
        input_rasters=input_rasters,
        output_labels=output_labels,
        res=res,
        min_res=min_res,
        max_res=max_res,
        crs=crs,
        res_tolerance=res_tolerance,
        max_upsampling=max_upsampling)
    pipeline = Compose(bricks=[brick])
    pipeline(folder)


@pytest.fixture(scope='module')
def get_file_high_res():
    folder = './tests/test_data/tmp1/'
    input_name = 'input_high'
    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'uint8',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.1, 0, 100000, 0, -0.1, 20000)}

    os.makedirs(folder, exist_ok=True)
    create_tiff_file(filename=os.path.join(folder, input_name + '.tif'), **profile)

    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def test_interpolation_custom(get_file_high_res):
    """
    testing if segmentation with mock adapter reproject the image
    input.res=(0.1, 0.1)
    """
    folder, name = get_file_high_res
    input_rasters = [name]
    output_labels = ['output1']
    folder = Path(folder)
    res = (0.5, 0.5)
    crs = 'EPSG:3857'
    min_res = 0.3
    max_res = 0.5

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16', min_res=min_res,
                                        max_res=max_res)
    with rasterio.open(folder/(output_labels[0] + '.tif')) as output:
        assert output.res[0] == res[0] and output.res[1] == res[1]
        assert output.crs == rasterio.crs.CRS.from_user_input(crs)


def test_interpolation_custom_src(get_file_high_res):
    """
    testing if segmentation with mock adapter do not change the resolution of the input image
    input.res=(0.1, 0.1)
    """
    folder, name = get_file_high_res
    input_rasters = [name]
    output_labels = ['output1']
    folder = Path(folder)
    res = (0.2, 0.2)
    crs = 'EPSG:3857'
    min_res = 0.1
    max_res = 0.4

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16', min_res=min_res,
                                        max_res=max_res)
    with rasterio.open(folder/(output_labels[0] + '.tif')) as output, rasterio.open(folder/(input_rasters[0] + '.tif')) as src:
        assert output.res[0] == src.res[0] and output.res[1] == src.res[1]
        assert output.crs == rasterio.crs.CRS.from_user_input(crs)


def test_interpolation_custom_error(get_file_high_res):
    """
    testing if segmentation with mock adapter do not change the resolution of the input image
    input.res=(0.1, 0.1)
    """
    folder, name = get_file_high_res
    input_rasters = [name]
    output_labels = ['output1']
    folder = Path(folder)
    res = (0.6, 0.6)
    crs = 'EPSG:3857'
    min_res = 0.1
    max_res = 0.4

    with pytest.raises(ValueError):
        _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                            input_dtype='uint16', output_dtype='uint16', min_res=min_res,
                                            max_res=max_res)


def test_interpolation(get_file):
    """
    testing if segmentation with mock adapter reproject the image
    input.res=(10.0, 10.0), dest_res=(10.0, 10.0)
    """
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output1']
    folder = Path(folder)
    res = (10.0, 10.0)
    crs = 'EPSG:3857'

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16')
    with rasterio.open(folder/(output_labels[0] + '.tif')) as output:
        assert output.res[0] == res[0] and output.res[1] == res[1]
        assert output.crs == rasterio.crs.CRS.from_user_input(crs)


def test_reproject_resample(get_file):
    """
    testing previous case with different resolution and CRS
    """
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output2']
    folder = Path(folder)
    res = (20, 20)
    crs = 'EPSG:32636'

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16')

    with rasterio.open(folder/(output_labels[0] + '.tif')) as output:
        assert output.res[0] == res[0] and output.res[1] == res[1]
        assert output.crs == rasterio.crs.CRS.from_user_input(crs)


def test_resample(get_file):
    """
    input.crs = 'EPSG:32636'
    testing, if passing dest_crs = input.crs doesn't alter CRS of output image, but only resamples the image
    """
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output3']
    folder = Path(folder)
    res = (5, 5)
    crs = 'EPSG:3857'

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16')

    with rasterio.open(folder/(output_labels[0] + '.tif')) as output:
        assert output.res[0] == res[0] and output.res[1] == res[1]
        assert output.crs == rasterio.crs.CRS.from_user_input(crs)


def test_max_upsampling(get_file):
    """
    Testing if ratio input.res / dest_res > max_upsampling, raises error
    """
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output3']
    folder = Path(folder)
    res = (2, 2)
    crs = 'EPSG:3857'

    with pytest.raises(ValueError):
        _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                            input_dtype='uint16', output_dtype='uint16')


def test_res_tolerance(get_file):
    """
    Testing if output resolution is within resolution tolerance, reprojection doesn't change resolution of input image
    res_tolerance=0.1
    input.res = (10.0, 10.0)
    input.crs = "EPSG:32636
    """
    # TO DO to write test
    folder, name = get_file
    input_rasters = [name]
    output_labels = ['output3']
    folder = Path(folder)
    res = (9.995, 9.995)
    crs = 'EPSG:3857'

    _run_segmentation_with_mock_adapter(input_rasters, output_labels, folder, res, crs,
                                        input_dtype='uint16', output_dtype='uint16')

    with rasterio.open(folder / (output_labels[0] + '.tif')) as output:
        with rasterio.open(folder / (input_rasters[0] + '.tif')) as src:
            # assert input resolution isn't changed
            assert output.res[0] == src.res[0] and output.res[1] == src.res[1]
            assert output.crs == rasterio.crs.CRS.from_user_input(crs)
