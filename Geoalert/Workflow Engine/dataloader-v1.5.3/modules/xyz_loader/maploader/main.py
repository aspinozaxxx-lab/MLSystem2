import os
import time
from typing import Optional, Tuple
import asyncio
import aiohttp
import shutil
from loguru import logger
from .merge import merge_tiles
from .tiles import get_tile
from .converters import get_converter
from .loader import Loader
from .extent import BBox, Extent


def generate_tiles(extent, zoom, projection, tile_size=256, nchannels=None):
    top, left, bottom, right = extent.bounds

    Tile = get_tile(projection)
    Converter = get_converter(projection)

    converter = Converter(zoom, tile_size)

    x_start = converter.lon_to_xtile(left)
    x_stop = converter.lon_to_xtile(right)

    y_start = converter.lat_to_ytile(top)
    y_stop = converter.lat_to_ytile(bottom)

    tiles = []

    for x in range(x_start, x_stop + 1):  # include last tile
        for y in range(y_start, y_stop + 1):  # include last tile

            tile = Tile(zoom, x, y, nchannels=nchannels)
            tiles.append(tile)

    if not isinstance(extent, BBox):
        tiles = [tile for tile in tiles if tile in extent]  # filter tiles outside extent

    return tiles


async def _load_tile(tile, loader, path):
    image = await loader.load_tile(tile)
    tile.save(path, image)


async def load_tiles(tiles, url, save_dir, source_type='xyz', header='default', credentials=None,
                     retry_attempts=10, retry_delay=2, response_timeout=10, rotate_agents=False,
                     tile_size=256, ignore_errors=False, proxy=None, connection_limit=10):
    if len(tiles) > connection_limit:
        logger.info(f'Loading with {connection_limit} connections')
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=connection_limit)) as session:
        loader = Loader(
            url,
            session=session,
            source_type=source_type,
            header=header,
            credentials=credentials,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            response_timeout=response_timeout,
            rotate_agents=rotate_agents,
            tile_size=tile_size,
            ignore_errors=ignore_errors,
            proxy=proxy,
            connection_limit=connection_limit
        )
        tasks = [asyncio.create_task(_load_tile(tile, loader, save_dir)) for tile in tiles]
        gather = asyncio.gather(*tasks, return_exceptions=False)
        try:
            await gather
        except Exception as e:
            print("Error " + str(e))
            gather.cancel()
            raise e


def download(
        url, zoom, aoi, output_fp,
        projection='epsg:3857',
        source_type='xyz',
        header=None,
        credentials: Optional[Tuple[str, str]] = None,
        retry_attempts=5,
        retry_delay=1,
        response_timeout=10,
        rotate_agents=False,
        tile_size=256,
        nchannels=None,
        delete_tiles=True,
        ignore_errors=False,
        compress_output=True,
        proxy=None,
        connection_limit=10,
        **kwargs
        ):
    """
    Load tiles by link and AOI.
    Args:
        url: z-x-y link for download
        zoom: zoom for loading
        aoi: GeoJSON geometry (Polygon), area of interests for loading
        output_fp: filename of output
        projection: one of epsg:3395, epsg:3857
        source_type: one of XYZ, TMS
        workers: int, number of processes for loading
        delay: time between requests, if workers=0
        header: request header
        credentials: request credentials
        retry_attempts: number of tries if tile loading failed
        retry_delay: delay between tries of tile loading failed
        response_timeout: requests timeout
        rotate_agents: if True, requests headers have different `user_agent`
        tile_size: default=256
        nchannels: default=None means that loader will try to make every tile to be RGB.
                Int value will force to download the tiles of a certain channel number. Intended to be 1,3 or 4
        delete_tiles: if True, delete tiles after loading and merging to TIFF
        ignore_errors: if True, in case of error "black" tile is generated

    """
    #
    ext = Extent(aoi)
    if credentials:
        credentials = aiohttp.BasicAuth(*credentials)
    # create directories and pathes
    base_dir = os.path.dirname(output_fp)
    tiles_dir = os.path.join(base_dir, '.tiles')
    os.makedirs(tiles_dir, exist_ok=True)

    # generate tiles
    tiles = generate_tiles(ext, zoom, projection, tile_size=tile_size, nchannels=nchannels)

    if tiles:
        # test one tile to avoid async simultaneous calls in case of 401 or 403 responses
        test_tiles = [tiles[0]]
        asyncio.run(load_tiles(
            test_tiles, url, tiles_dir,
            source_type=source_type,
            header=header,
            credentials=credentials,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            response_timeout=response_timeout,
            rotate_agents=rotate_agents,
            tile_size=tile_size,
            ignore_errors=ignore_errors,
            proxy=proxy,
            connection_limit=connection_limit
        ))
        logger.debug("Test tile loaded successfully, proceeding to download the rest of tiles")

    begin = time.time()
    # load tiles
    asyncio.run(load_tiles(
        tiles, url, tiles_dir,
        source_type=source_type,
        header=header,
        credentials=credentials,
        retry_attempts=retry_attempts,
        retry_delay=retry_delay,
        response_timeout=response_timeout,
        rotate_agents=rotate_agents,
        tile_size=tile_size,
        ignore_errors=ignore_errors,
        proxy=proxy,
        connection_limit=connection_limit
    ))
    logger.info(f"Loaded {len(tiles)} tiles from {url} at {zoom} zoom in {(time.time() - begin):.2f} seconds")

    begin = time.time()
    merge_tiles(tiles, tiles_dir, output_fp, compress=compress_output)
    logger.debug(f"Merged {len(tiles)} tiles in {(time.time() - begin):.2f} seconds")

    # remove saved tiles after assembling whole image
    if delete_tiles:
        shutil.rmtree(tiles_dir)
