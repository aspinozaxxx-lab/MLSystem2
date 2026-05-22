import pytest
import asyncio
import sys
from loguru import logger
from time import time
from maploader.main import load_tiles
from maploader.tiles import WebTile
from maploader.errors import TileNotLoaded

sleep_url = 'http://127.0.0.1:8000/tile/{sleep}'  # /{z}/{x}/{y}'


LOG_LEVEL = "INFO"
logger.remove()
logger.add(sys.stdout, level=LOG_LEVEL, format="{time:YYYY-MM-DD-HH:mm:ss} {level} {message}")


def test_load_unauthorized_exits_instantly(test_data):
    """
        Look to the server logs: must be exactly one 401 response
    """
    tiles = [WebTile(1, 401, 1),
             WebTile(1, 401, 2),
             WebTile(1, 401, 3),
             WebTile(1, 401, 4),
             WebTile(1, 401, 5),
             WebTile(1, 401, 6)]
    url = sleep_url.format(sleep=1) + '/{z}/{x}/{y}'
    with pytest.raises(TileNotLoaded) as e:
        asyncio.run(load_tiles([tiles[0]], url=url, save_dir=test_data, retry_delay=0.1, retry_attempts=2,
                               ignore_errors=False))
        asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=0.1, retry_attempts=2,
                               ignore_errors=False))
    assert 401 == e.value.parameters['status']


def test_load_unauthorized_sends_multiple_requests(test_data):
    """
    Improper behavior, the real load_tiles works as above test
    Look to the server logs: must be several 403 responses
    """

    url = sleep_url.format(sleep=1) + '/{z}/{x}/{y}'
    tiles = [WebTile(1, 403, 1),
             WebTile(1, 403, 2),
             WebTile(1, 403, 3),
             WebTile(1, 403, 4),
             WebTile(1, 403, 5),
             WebTile(1, 403, 6),
             WebTile(1, 403, 7),
             WebTile(1, 403, 8),
             WebTile(1, 403, 9)]
    with pytest.raises(TileNotLoaded) as e:
        asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=0.1, retry_attempts=2,
                               ignore_errors=False))
    assert 403 == e.value.parameters['status']


def test_retry_delay(test_data):
    url = sleep_url.format(sleep=0) + '/{z}/{x}/{y}'
    tiles = [WebTile(0, 404, 0)]
    begin = time()
    with pytest.raises(TileNotLoaded) as e:
        asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=1, retry_attempts=5,
                               ignore_errors=False))
    assert 404 == e.value.parameters['status']
    # estimated time is 1+2+4+8+16=31
    assert 30 < time() - begin < 33

    begin = time()
    with pytest.raises(TileNotLoaded) as e:
        asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=2, retry_attempts=1,
                               ignore_errors=False))
    assert 404 == e.value.parameters['status']
    # estimated time is 2
    assert 2 < time() - begin < 3
    '''
    Test that the retry delay is growing each retry
    '''


def test_connection_number(test_data):
    url = sleep_url.format(sleep=1) + '/{z}/{x}/{y}'
    tiles = [WebTile(18, 200, y) for y in range(50)]
    begin = time()
    asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=1, retry_attempts=5,
                           ignore_errors=False, connection_limit=5))
    # 50 tiles with 5 connections = 10+ seconds
    logger.info(f"WITH 4 connections: {time() - begin}")
    assert 10 < time() - begin < 12

    begin = time()
    asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=1, retry_attempts=5,
                           ignore_errors=False, connection_limit=10))
    logger.info(f"WITH 10 connections: {time() - begin}")
    assert 5 < time() - begin < 7

    begin = time()
    asyncio.run(load_tiles(tiles, url=url, save_dir=test_data, retry_delay=1, retry_attempts=5,
                           ignore_errors=False, connection_limit=50))
    logger.info(f"WITH 50 connections: {time() - begin}")
    assert 1 < time() - begin < 2
