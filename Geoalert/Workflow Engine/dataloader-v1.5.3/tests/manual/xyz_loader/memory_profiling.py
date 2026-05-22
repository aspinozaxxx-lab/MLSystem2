import os
import pytest
import numpy as np
from maploader.tiles import WebTile
from maploader.merge import merge_tiles

sleep_url = 'http://127.0.0.1:8000/tile/{sleep}'  # /{z}/{x}/{y}'

# The merged area is SIZE*SIZE tiles
SIZE = 30


@pytest.fixture
def generate_tiles():
    test_data_dir = './tests/test_data'
    os.makedirs(test_data_dir)
    z = 18
    size = SIZE
    tiles = [WebTile(z, 1000+x, y) for x in range(size) for y in range(size)]
    for n, tile in enumerate(tiles):
        tile.save(test_data_dir, np.ones(shape=(3, 256, 256), dtype='uint8')*(n%255))

    yield tiles, test_data_dir

    for file in os.listdir(test_data_dir):
        try:
            os.remove(os.path.join(test_data_dir, file))
        except:
            pass
    try:
        os.rmdir(test_data_dir)
    except:
        pass

"""

Function is commented to not run in general `test.sh manual` command, use `test.sh memory` to run it

def test_mem(generate_tiles):
    '''
    Use `docker stats` to monitor consumptioin of memory
    See pytest output to calculate time
    '''
    tiles, datadir = generate_tiles
    merge_tiles(tiles, datadir, datadir+'/out.tif')

"""
