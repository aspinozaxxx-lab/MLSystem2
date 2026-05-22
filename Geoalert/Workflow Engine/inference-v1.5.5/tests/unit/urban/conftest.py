import os
import shutil
import pytest
from .generate_files import create_tiff_file


@pytest.fixture(scope='session')
def get_file():
    folder = './tests/test_data/tmp/'
    name = 'B08'
    profile = {'height': 2048,
               'width': 2048,
               'count': 1,
               'dtype': 'uint16',
               'nodata': 0,
               'crs': 'EPSG:3857',
               'transform': (10.0, 0, 100000, 0, -10.0, 20000)}

    os.makedirs(folder, exist_ok=True)
    create_tiff_file(filename=os.path.join(folder, name + '.tif'), **profile)

    yield folder, name

    try:
        shutil.rmtree(folder)
    except OSError:
        pass
