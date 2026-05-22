import os
import shutil
import rasterio
import shapely
import numpy as np
import pytest
from pathlib import Path
from urban import Compose, ZonalStats, ZonalMedian
from urban.functional import io
from gpdadapter import FeatureCollection
from .generate_files import create_tiff_file


@pytest.fixture(scope='module')
def get_raster_vector():
    folder = './tests/test_data/tmp1/'
    raster_name = 'zonal_raster'
    vector_name = 'zonal_vector'

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


def test_zonal_median(get_raster_vector):
    folder, raster_name, vector_name, max_height, min_height = get_raster_vector
    output_name = 'output_median'
    folder = Path(folder)

    pipeline = Compose(bricks=[ZonalMedian(input_raster=raster_name, input_vector=vector_name, field_name='median',
                                           output_vector=output_name)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 2
    assert output_fc[0, 'median'] == max_height
    assert output_fc[1, 'median'] == min_height


def test_zonal_min(get_raster_vector):
    folder, raster_name, vector_name, max_height, min_height = get_raster_vector
    output_name = 'output_min'
    folder = Path(folder)

    pipeline = Compose(bricks=[ZonalStats(input_raster=raster_name, input_vector=vector_name, statistics=['min'],
                                          output_vector=output_name)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 2
    assert output_fc[0, 'min'] == min_height
    assert output_fc[1, 'min'] == min_height


def test_zonal_max(get_raster_vector):
    folder, raster_name, vector_name, max_height, min_height = get_raster_vector
    output_name = 'output_min'
    folder = Path(folder)
    nodata_value = 0

    pipeline = Compose(bricks=[ZonalStats(input_raster=raster_name, input_vector=vector_name, statistics=['max'],
                                          output_vector=output_name, nodata_value=nodata_value)])
    pipeline(folder)

    output_fc = FeatureCollection.from_file(folder / (output_name + '.geojson'))

    assert len(output_fc) == 2
    assert output_fc[0, 'max'] == max_height
    assert output_fc[1, 'max'] == min_height
