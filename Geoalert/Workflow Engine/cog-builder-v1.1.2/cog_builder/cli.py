import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from .functional.build_cog import CogBuilder
from .functional.geometry import maybe_valid_geometry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'COG Builder',
        description = 'Build Cloud-Optimized GeoTIFF from regular GeoTIFF',
        epilog = 'Get it and use it well')

    parser.add_argument('raster_uri', help="Input GeoTIFF file name")
    parser.add_argument('target_uri', help="Result COG file name")
    parser.add_argument('--aoi', action='store', dest='mask_geometry',
                        help="GeoJSON geometry string used crop GeoTIFF, conaining a `geometry` object "
                             "like {'type': 'Polygon', 'coordinates': [[[...],[...],...]] }")
    parser.add_argument('--channels', action='store', default='1,2,3', help="Raster bands to be treated as RGB, starting from 1")
    parser.add_argument('--compress', action='store', default='WEBP', help = "Compression method for output COG. Use WEBP for the best and compact results, ZSTD for losless or JPEG for compatibility")
    args = parser.parse_args()

    if args.mask_geometry:
        geom = maybe_valid_geometry(json.loads(args.mask_geometry))
    else:
        geom = None

    with TemporaryDirectory() as tempdir:
        CogBuilder(input_ds=Path(args.raster_uri),
                   cog_ds=Path(args.target_uri),
                   workdir=Path(tempdir),
                   channels=args.channels.split(","),
                   aoi=geom,
                   compress=args.compress)
