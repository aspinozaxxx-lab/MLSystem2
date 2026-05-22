from shapely.geometry import shape
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from rasterio.features import bounds


def polygon_from_bounds(top, left, bottom, right):
    geometry = {
                'type': 'Polygon',
                'coordinates': [[[left, top],
                                 [right, top],
                                 [right, bottom],
                                 [left, bottom],
                                 [left, top]]]
        }

    return geometry


def shape_from_bounds(top, left, bottom, right):
    return shape(polygon_from_bounds(top, left, bottom, right))


def to_quad_key(x: int, y: int, z: int) -> str:
    """Convert x,y,z coords to quadkey
    https://docs.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system?redirectedfrom=MSDN
    """
    index = ""
    for i in reversed(range(1, z + 1)):
        b = 0
        mask = 1 << (i - 1)
        if x & mask != 0:
            b += 1
        if y & mask != 0:
            b += 2
        index += str(b)
    return index
