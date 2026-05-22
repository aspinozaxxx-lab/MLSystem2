import os
import shutil
import numpy as np
import rasterio
import shapely
import pytest
from pathlib import Path
from urban import Compose, CrownDelineation, CrownMaxHeight
from gpdadapter import FeatureCollection
from urban.functional import io
from .generate_files import create_tiff_file


@pytest.fixture(scope='module')
def get_chm_file():
    folder = './tests/test_data/tmp1/'
    input_name = 'chm'
    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.6, 0, 100000, 0, -0.6, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = profile['height'] // 2
    x_center = profile['width'] // 2
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = 0
    data[:, :, x_center - epsilon:x_center + epsilon] = 0
    create_tiff_file(filename=os.path.join(folder, input_name + '.tif'), data=data, **profile)

    yield folder, input_name, max_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_file_high_res():
    folder = './tests/test_data/tmp1/'
    input_name = 'chm'
    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.000005, 0, 100000, 0, -0.000005, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = profile['height'] // 2
    x_center = profile['width'] // 2
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = 0
    data[:, :, x_center - epsilon:x_center + epsilon] = 0
    create_tiff_file(filename=os.path.join(folder, input_name + '.tif'), data=data, **profile)

    yield folder, input_name, max_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_file_low_res():
    folder = './tests/test_data/tmp1/'
    input_name = 'chm'
    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (30.0, 0, 100000, 0, -30.0, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = profile['height'] // 2
    x_center = profile['width'] // 2
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = 0
    data[:, :, x_center - epsilon:x_center + epsilon] = 0
    create_tiff_file(filename=os.path.join(folder, input_name + '.tif'), data=data, **profile)

    yield folder, input_name, max_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_file_empty():
    folder = './tests/test_data/tmp1/'
    input_name = 'chm'
    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.6, 0, 100000, 0, -0.6, 20000)}

    os.makedirs(folder, exist_ok=True)
    create_tiff_file(filename=os.path.join(folder, input_name + '.tif'), matrix='zeros', **profile)

    yield folder, input_name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_raster_vector():
    folder = './tests/test_data/tmp1/'
    raster_name = 'chm_raster'
    vector_name = 'chm_vector'

    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.6, 0, 100000, 0, -0.6, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    min_height = 5.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = 75
    x_center = 100
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = min_height
    data[:, :, x_center - epsilon:x_center + epsilon] = min_height
    create_tiff_file(filename=os.path.join(folder, raster_name + '.tif'), data=data, **profile)

    with rasterio.open(os.path.join(folder, raster_name + '.tif')) as src:
        transform = src.transform
        crs = src.crs

    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 30], [30, 30], [30, 0]]),
                                         shapely.Polygon([[100, 0], [100, 75], [145, 75], [145, 0]])],
                            'confidence': [1, 0.5]},
                           crs=crs)
    fc[0, 'geometry'] = shapely.affinity.affine_transform(fc[0, 'geometry'], transform.to_shapely())
    fc[1, 'geometry'] = shapely.affinity.affine_transform(fc[1, 'geometry'], transform.to_shapely())
    io.save_fc(fc, path=folder, name=vector_name)

    yield folder, raster_name, vector_name, max_height, min_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_raster_vector_out():
    folder = './tests/test_data/tmp1/'
    raster_name = 'chm_raster'
    vector_name = 'chm_vector'

    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.6, 0, 100000, 0, -0.6, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    min_height = 5.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = 75
    x_center = 100
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = min_height
    data[:, :, x_center - epsilon:x_center + epsilon] = min_height
    create_tiff_file(filename=os.path.join(folder, raster_name + '.tif'), data=data, **profile)

    with rasterio.open(os.path.join(folder, raster_name + '.tif')) as src:
        transform = src.transform
        crs = src.crs

    fc = FeatureCollection({'geometry': [shapely.Polygon([[0, 0], [0, 30], [30, 30], [30, 0]]),
                                         shapely.Polygon([[1000, 0], [1000, 750], [1450, 750], [1450, 0]])],
                            'confidence': [1, 0.5]},
                           crs=crs)
    fc[0, 'geometry'] = shapely.affinity.affine_transform(fc[0, 'geometry'], transform.to_shapely())
    fc[1, 'geometry'] = shapely.affinity.affine_transform(fc[1, 'geometry'], transform.to_shapely())
    io.save_fc(fc, path=folder, name=vector_name)

    yield folder, raster_name, vector_name, max_height, min_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


@pytest.fixture(scope='module')
def get_chm_raster_vector_empty():
    folder = './tests/test_data/tmp1/'
    raster_name = 'chm_raster'
    vector_name = 'chm_vector'

    profile = {'height': 150,
               'width': 200,
               'count': 1,
               'dtype': 'float32',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (0.6, 0, 100000, 0, -0.6, 20000)}

    os.makedirs(folder, exist_ok=True)
    max_height = 25.0
    min_height = 5.0
    data = np.ones(shape=(profile['count'], profile['height'], profile['width']), dtype=profile['dtype']) * max_height
    y_center = 75
    x_center = 100
    epsilon = 50
    data[:, y_center - epsilon:y_center + epsilon, :] = min_height
    data[:, :, x_center - epsilon:x_center + epsilon] = min_height
    create_tiff_file(filename=os.path.join(folder, raster_name + '.tif'), data=data, **profile)

    fc = FeatureCollection()
    io.save_fc(fc, path=folder, name=vector_name)

    yield folder, raster_name, vector_name, max_height, min_height

    try:
        shutil.rmtree(folder)
    except OSError:
        pass


def test_crown_delineation(get_chm_file):
    folder, input_name, max_height = get_chm_file
    output_name = 'output'
    folder = Path(folder)

    pipeline = Compose(bricks=[CrownDelineation(input_raster=input_name, output_vector=output_name)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 4
    assert output_fc[0, 'max_height'] == max_height
    assert output_fc[1, 'max_height'] == max_height
    assert output_fc[2, 'max_height'] == max_height
    assert output_fc[3, 'max_height'] == max_height


def test_crown_delineation_high_res(get_chm_file_high_res):
    folder, input_name, max_height = get_chm_file_high_res
    output_name = 'output_high'
    folder = Path(folder)

    pipeline = Compose(bricks=[CrownDelineation(input_raster=input_name,
                                                output_vector=output_name,
                                                max_pixel_number=50)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 4
    assert output_fc[0, 'max_height'] == max_height
    assert output_fc[1, 'max_height'] == max_height
    assert output_fc[2, 'max_height'] == max_height
    assert output_fc[3, 'max_height'] == max_height


def test_crown_delineation_low_res(get_chm_file_low_res):
    folder, input_name, max_height = get_chm_file_low_res
    output_name = 'output_low'
    folder = Path(folder)

    pipeline = Compose(bricks=[CrownDelineation(input_raster=input_name, output_vector=output_name)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 4
    assert output_fc[0, 'max_height'] == max_height
    assert output_fc[1, 'max_height'] == max_height
    assert output_fc[2, 'max_height'] == max_height
    assert output_fc[3, 'max_height'] == max_height


def test_crown_delineation_empty(get_chm_file_empty):
    folder, input_name = get_chm_file_empty
    output_name = 'output_empty'
    folder = Path(folder)

    pipeline = Compose(bricks=[CrownDelineation(input_raster=input_name, output_vector=output_name)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 0


def test_crown_max_height(get_chm_raster_vector):
    folder, raster_name, vector_name, max_height, min_height = get_chm_raster_vector
    output_name = 'output_crown_max_height'
    folder = Path(folder)
    min_height_for_tree = 6
    pipeline = Compose(bricks=[CrownMaxHeight(input_raster=raster_name, input_vector=vector_name,
                                              output_vector=output_name, min_height_for_tree=min_height_for_tree)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 1
    assert output_fc[0, 'max_height'] == max_height


def test_crown_max_height_out(get_chm_raster_vector_out):
    folder, raster_name, vector_name, max_height, min_height = get_chm_raster_vector_out
    output_name = 'output_crown_max_height_out'
    folder = Path(folder)
    min_height_for_tree = 6
    pipeline = Compose(bricks=[CrownMaxHeight(input_raster=raster_name, input_vector=vector_name,
                                              output_vector=output_name, min_height_for_tree=min_height_for_tree)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 1
    assert output_fc[0, 'max_height'] == max_height


def test_crown_max_height_empty(get_chm_raster_vector_empty):
    folder, raster_name, vector_name, max_height, min_height = get_chm_raster_vector_empty
    output_name = 'output_crown_max_height_out'
    folder = Path(folder)
    min_height_for_tree = 6
    pipeline = Compose(bricks=[CrownMaxHeight(input_raster=raster_name, input_vector=vector_name,
                                              output_vector=output_name, min_height_for_tree=min_height_for_tree)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 0
