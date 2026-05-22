import pytest
import asyncio
import aiohttp
import numpy as np
from maploader.loader import Loader
from maploader.tiles import WebTile
from aioresponses import aioresponses
from maploader.errors import TileNotLoaded, TileNotReadable

mock_url = 'https://tileserver.url/{z}/{x}/{y}.png'
tile_location = {'x': 1, 'y': 1, 'z': 1}
login = 'login'
password = 
basic_auth = 'bG9naW46cGFzc3dvcmQ='


async def _load_tile(tile, **kwargs):
    async with aiohttp.ClientSession(trust_env=True) as session:
        loader = Loader(session=session, **kwargs)
        image = await loader.load_tile(tile)
    return image


@pytest.mark.asyncio
async def test_returns_empty_tile_on_no_content():
    with aioresponses() as mocked:
        mocked.get(mock_url.format(**tile_location), status=204, body=b"")
        tile = WebTile(**tile_location)
        res = await _load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2, ignore_errors=False)
    assert isinstance(res, np.ndarray) and res.dtype == np.uint8 and res.max() == 0


@pytest.mark.asyncio
async def test_ignores_error():
    with aioresponses() as mocked:
        mocked.get(mock_url.format(**tile_location), status=404, body=b"")
        tile = WebTile(**tile_location)
        res = await _load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2, ignore_errors=True)
    assert isinstance(res, np.ndarray) and res.dtype == np.uint8 and res.max() == 0


def test_raises_exception_with_correct_code():
    tile = WebTile(**tile_location)
    for code in (404, 500):
        with aioresponses() as mocked:
            mocked.get(mock_url.format(**tile_location), status=code, body=b"", repeat=True)
            with pytest.raises(TileNotLoaded) as e:
                asyncio.run(_load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2,
                                       ignore_errors=False))
            assert code == e.value.parameters['status']


def test_raises_unauthorized_or_forbidden_error_without_retry():
    tile = WebTile(**tile_location)
    for code in (401, 403):
        with aioresponses() as mocked:
            # repeat=False is important: if the loader retries download,
            # the mock server will not answer and it will be another exception
            mocked.get(mock_url.format(**tile_location), status=code, body=b"", repeat=False)
            with pytest.raises(TileNotLoaded) as e:
                asyncio.run(_load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2,
                                       ignore_errors=False))
            assert code == e.value.parameters['status']


def test_raises_decode_error_exception():
    tile = WebTile(**tile_location)
    with aioresponses() as mocked:
        mocked.get(mock_url.format(**tile_location), status=200, body=b"", repeat=True)
        with pytest.raises(TileNotReadable) as e:
            asyncio.run(_load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2, ignore_errors=False))


def test_returns_tile(get_tile):
    test_image = get_tile
    with aioresponses() as mocked:
        mocked.get(mock_url.format(**tile_location), status=200, body=open(test_image, 'rb').read())
        tile = WebTile(**tile_location)
        res = asyncio.run(_load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2))
        assert isinstance(res, np.ndarray) \
               and res.shape == (256, 256, 3) \
               and res.max() > 0

'''
def test_auth_no_creds():
    with aioresponses() as mocked:
        headers = {'Authorization': 'Basic ' + basic_auth}
        mocked.get(mock_url.format(**tile_location), status=404, body=open(test_image, 'rb').read(), headers=headers)
        tile = WebTile(**tile_location)
        #with pytest.raises(TileNotLoaded) as e:
        res = asyncio.run(_load_tile(tile, url=mock_url, retry_delay=0.1, retry_attempts=2))
        assert isinstance(res, np.ndarray)

'''
