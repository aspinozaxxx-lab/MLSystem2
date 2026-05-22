import os
import rasterio
from osgeo import gdal
from typing import List
from .tiles import Tile
from affine import Affine
from rasterio.windows import Window


def get_profile(tiles: List[Tile]):
    tiles.sort(key=lambda tile: tile.x)
    left_tile = tiles[0]
    right_tile = tiles[-1]

    tiles.sort(key=lambda tile: tile.y)
    top_tile = tiles[0]
    bottom_tile = tiles[-1]

    xmin = left_tile.x
    xmax = right_tile.x
    ymin = top_tile.y
    ymax = bottom_tile.y

    x, _ = left_tile.coords
    _, y = top_tile.coords

    transform = Affine(left_tile.gsd, 0, x, 0, - left_tile.gsd, y)

    width = left_tile.size * (xmax-xmin+1)
    height = left_tile.size * (ymax-ymin+1)
    return {'width': width,
            'height': height,
            'transform': transform,
            'count': left_tile.nchannels or 3,
            'crs': left_tile.crs,
            'dtype': 'uint8',
            'driver': 'GTiff'}, xmin, ymin


def merge_tiles(tiles: List[Tile],
                tiles_dir,
                output_fp,
                compress=True):
    profile, xmin, ymin = get_profile(tiles)
    size = tiles[0].size
    tmp = os.path.join(tiles_dir, 'tmp.tif')
    # Writing tiles one-by-ones is more memory-efficient than using rasterio.merge
    with rasterio.open(tmp, 'w', **profile) as dst:
        for tile in tiles:
            data = tile.read(tiles_dir)
            window = Window((tile.x - xmin)*size, (tile.y-ymin)*size, size, size)
            dst.write(data, window=window)

    if compress:
        # Forming COG in a separate stage by GDAL because WEBP driver is very slow with tile-by-tile writing
        translate_opts = f"-of COG -co TILING_SCHEME=GoogleMapsCompatible" \
                         f" -co COMPRESS=ZSTD -co LEVEL=2 -co PREDICTOR=2 -co BIGTIFF=IF_SAFER -co NUM_THREADS=2 "
        gdal.Translate(output_fp, tmp, options=translate_opts)
        os.remove(tmp)
    else:
        os.rename(tmp, output_fp)
