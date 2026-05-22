import json
from shapely.geometry import shape
from .utils import polygon_from_bounds


def read_geojson(fp):

    with open(fp) as f:
        gj = json.load(f)

    exts = []
    for feature in gj['features']:
        geometry = feature['geometry']
        ext = Extent(geometry)
        exts.append(ext)
    return exts


class Extent:

    def __init__(self, geometry):
        self.geometry = geometry
        self._shape = shape(self.geometry).buffer(0)

    @property
    def shape(self):
        return self._shape

    def __contains__(self, tile):
        return not tile.intersection(self._shape).is_empty

    @property
    def bounds(self):
        left, bottom, right, top = self._shape.bounds
        return top, left, bottom, right


class BBox(Extent):

    def __init__(self, top, left, bottom, right):

        geometry = polygon_from_bounds(top, left, bottom, right)
        super().__init__(geometry)
