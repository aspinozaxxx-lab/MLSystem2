import os
import pytest
import numpy as np
from skimage.io import imsave


# ================== pytest fixtures =============================
@pytest.fixture(scope='session')
def get_tile():
    tile_dir = './tests/test_tile'
    test_tile = tile_dir + '/tile.png'
    os.makedirs(tile_dir)
    imsave(test_tile, np.ones(shape=(256, 256), dtype='uint8'))

    yield test_tile

    try:
        os.remove(test_tile)
        os.rmdir(tile_dir)
    except OSError:
        pass


@pytest.fixture
def test_data():
    test_data_dir = './tests/test_data'
    os.makedirs(test_data_dir, exist_ok=True)
    yield test_data_dir

    for file in os.listdir(test_data_dir):
        try:
            os.remove(os.path.join(test_data_dir, file))
        except:
            pass
    try:
        os.rmdir(test_data_dir)
    except:
        pass
