import pytest

from maploader.errors import (MaploaderError,
                              MaploaderInternalError,
                              WrongChannelsNum,
                              WrongTileSize,
                              WrongSourceType,
                              WrongNdim,
                              TileNotLoaded,
                              CrsIsNotSupported)


def test_init_raises_assertion_error_on_insufficient_params():
    with pytest.raises(AssertionError):
        raise MaploaderError(message="Message with placeholder {param}")
    with pytest.raises(AssertionError):
        raise MaploaderError(message="Message with two placeholders {param1} {param2}", param1=0)


def test_init_maploader_internal_error():
    with pytest.raises(MaploaderError):
        raise MaploaderInternalError(error_message='Test error message')


def test_init_wrong_channel_num():
    with pytest.raises(MaploaderError):
        raise WrongChannelsNum(3,1)


def test_init_wrong_ndim():
    with pytest.raises(MaploaderError):
        raise WrongNdim(5)


def test_init_wrong_tile_size():
    with pytest.raises(MaploaderError):
        raise WrongTileSize(expected_size=256, real_size=512)


def test_init_wrong_source_type():
    with pytest.raises(MaploaderError):
        raise WrongSourceType('abcdef')


def test_init_tile_not_loaded():
    with pytest.raises(MaploaderError) as e:
        raise TileNotLoaded(tile_location='https://tile.server.com/18/100001/200002?token=,
                            proxy=None,
                            exception_message='Tile not loaded',
                            http_status=404)
    assert e.value.parameters['tile_location'] == 'https://tile.server.com/18/100001/200002'


def test_init_crs_not_supported():
    with pytest.raises(MaploaderError):
        raise CrsIsNotSupported(supported_projections=['EPSG:3857', 'EPSG:3395'], real_projection='EPSG:4326')


