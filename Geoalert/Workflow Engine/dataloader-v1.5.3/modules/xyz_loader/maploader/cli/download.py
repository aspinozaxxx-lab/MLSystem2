from loguru import logger
import os
import random
import shutil
import string
import asyncio

import click

from ..extent import BBox, read_geojson
from ..main import generate_tiles, load_tiles, merge_tiles


def get_random_folder():
    letters = string.ascii_lowercase
    return './.' + ''.join(random.choice(letters) for _ in range(10))


@click.command()
@click.argument('url')
@click.option('-z', '--zoom', default=14, show_default=True, type=int, help='Source zoom')
@click.option('-b', '--bbox', nargs=4, type=float, help='Coordinates of area bounding box in format (top, left, bottom, right)')
@click.option('-e', '--extent', type=str, help='Path to GeoJSON with extent')
@click.option('-o', '--output', required=True, type=str, help='Path to output raster')
@click.option('-c', '--crs', default='epsg:3857', show_default=True, help='CRS one of EPSG:3395 or EPSG:3857')
@click.option('-t', '--timeout', default=0.5, show_default=True, type=float, help='Timeout between tiles requests')
@click.option('-s', '--tile_size', default=256, show_default=True, type=int, help='Tile size in pixels')
@click.option('-n', '--nchannels', default=0, show_default=True, type=int, help='Tile channels number in pixels')
@click.option('-i', '--tiles_dir', default=None,  type=str, help='Directory to store tiles, default is random generated path')
@click.option('-d', '--delete_tiles', is_flag=True, help='Flag ot delete tiles dir after downloading and merging')
@click.option('-w', '--workers', default=0, help='If workers > 0, loading in async mode without delay')
@click.option('-u', '--user', help='User name for authentication')
@click.option('-p', '--password', help='Password for authentication')
def download(
    url,
    zoom,
    bbox,
    extent,
    output,
    crs,
    timeout,
    tile_size,
    nchannels,
    tiles_dir,
    delete_tiles,
    workers,
    user,
    password,
     ):

    # define area of interests
    if bbox is not None:
        ext = BBox(*bbox)
    else:
        ext = read_geojson(extent)[0]

    # generate loading tiles
    tiles = generate_tiles(ext, zoom, crs, tile_size, nchannels=None if nchannels == 0 else nchannels)
    logger.info('Generated {} tile for loading...'.format(len(tiles)))

    # start loading tiles
    if tiles_dir is None:
        tiles_dir = get_random_folder()
        os.makedirs(tiles_dir, exist_ok=True)
        logger.info('Created intermediate directory for tiles: {}'.format(tiles_dir))

    logger.info('Start loading tiles...')
    if user is not None:
        credentials = (user, password)
    else:
        credentials = None
    asyncio.run(load_tiles(tiles, url, tiles_dir, credentials=credentials))

    # merging tiles
    logger.info('Start merging tiles...')
    merge_tiles(tiles, tiles_dir, output)
    logger.info('Output have been saved: {}'.format(output))

    if delete_tiles:
        logger.info('Deleting intermediate directory with tiles: {}'.format(tiles_dir))
        shutil.rmtree(tiles_dir)

    logger.info('Done!')
