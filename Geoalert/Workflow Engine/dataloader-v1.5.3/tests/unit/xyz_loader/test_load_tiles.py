import pytest
import asyncio
from time import sleep
from maploader.main import load_tiles
from maploader.tiles import WebTile
from aioresponses import aioresponses, CallbackResult
from maploader.errors import TileNotLoaded

mock_url = 'https://tileserver.url/{z}/{x}/{y}.png'
tile_location = {'x': 1, 'y': 1, 'z': 1}
login = 'login'
password = 
basic_auth = 'bG9naW46cGFzc3dvcmQ='


async def wait_and_unauthorize(url, **kwargs):
    # sleep to allow another connections to emerge, if there are any
    await asyncio.sleep(1)
    return CallbackResult(status=401, reason="Unauthorized")


async def wait_and_reject(url, **kwargs):
    # sleep to allow another connections to emerge, if there are any
    await asyncio.sleep(1)
    return CallbackResult(status=403, reason="Forbidden")


def test_load_unauthorized_exits_instantly():
    tiles = [WebTile(**tile_location)]*10
    with aioresponses() as mocked:
        # repeat=False is important: if the loader retries download,
        # the mock server will not answer and it will be another exception
        mocked.get(mock_url.format(**tile_location), body=b"", repeat=False, callback=wait_and_unauthorize)
        mocked.get(mock_url.format(**tile_location), body=b"", repeat=True, callback=wait_and_reject)
        with pytest.raises(TileNotLoaded) as e:
            asyncio.run(load_tiles(tiles, url=mock_url, save_dir='/tmp', retry_delay=0.1, retry_attempts=2,
                                   ignore_errors=False))
        assert 401 == e.value.parameters['status']


def test_load_after_retry(get_tile):
    test_image = get_tile
    tiles = [WebTile(**tile_location)]*10
    with aioresponses() as mocked:
        # first attempt will get error, but after retry it must be successful
        mocked.get(mock_url.format(**tile_location), body=b"", repeat=False, status=500)
        mocked.get(mock_url.format(**tile_location), status=200, body=open(test_image, 'rb').read(), repeat=True)

        asyncio.run(load_tiles(tiles, url=mock_url, save_dir='/tmp', retry_delay=0.1, retry_attempts=2,
                                   ignore_errors=False))


def test_retry_limit(get_tile):
    test_image = get_tile
    tiles = [WebTile(**tile_location)]
    with aioresponses() as mocked:
        # first attempt will get error, but after retry it must be successful
        mocked.get(mock_url.format(**tile_location), body=b"", repeat=False, status=500)
        mocked.get(mock_url.format(**tile_location), body=b"", repeat=False, status=503)
        mocked.get(mock_url.format(**tile_location), status=200, body=open(test_image, 'rb').read(), repeat=True)
        with pytest.raises(TileNotLoaded) as e:
            asyncio.run(load_tiles(tiles, url=mock_url, save_dir='/tmp', retry_delay=0.1, retry_attempts=1,
                                       ignore_errors=False))
        assert 503 == e.value.parameters['status']