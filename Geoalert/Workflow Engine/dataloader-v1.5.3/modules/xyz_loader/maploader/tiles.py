import os
import math
from pyproj import Proj
import rasterio
from scipy.optimize import fsolve

from .io import read_image, save_image
from .converters import WorldConverter
from .utils import shape_from_bounds
from .errors import CrsIsNotSupported


def get_tile(projection):

    converters = {
        'epsg:3857': WebTile,
        'epsg:3395': WorldTile,
    }
    projection = projection.lower()
    if projection in converters.keys():
        c = converters[projection]
    else:
        raise CrsIsNotSupported(list(converters.keys()), projection)

    return c


class Tile:

    crs = None
    proj = None

    def __init__(self, z, x, y, size=256, nchannels=None):
        self.z = z
        self.x = x
        self.y = y
        self.size = size
        self.nchannels = nchannels

    def __repr__(self):
        return '<Tile x={} y={} z={} s={}px n={}>'.format(self.x, self.y, self.z, self.size, self.nchannels)

    @property
    def name(self):
        return '{z}_{x}_{y}'.format(
            z=self.z,
            x=self.x,
            y=self.y,
        )

    @classmethod
    def _next_tile(cls, self):
        return cls(self.z, self.x + 1, self.y + 1, self.size)

    def read(self, dir):
        path = os.path.join(dir, self.name + '.tif')
        return read_image(path)

    def save(self, dir, image):
        path = os.path.join(dir, self.name + '.tif')
        save_image(path, image, crs=self.crs, transform=self.transform)

    def open(self, dir):
        path = os.path.join(dir, self.name + '.tif')
        return rasterio.open(path)

    @property
    def transform(self):
        x, y = self.coords
        return [self.gsd, 0, x, 0, - self.gsd, y]

    @property
    def shape(self):
        return shape_from_bounds(*self.bounds)

    @property
    def bounds(self):
        next_tile = self._next_tile(self)
        top = self.lat
        left = self.lon
        bottom = next_tile.lat
        right = next_tile.lon
        return top, left, bottom, right

    def intersection(self, obj):
        return self.shape.intersection(obj)

    @property
    def lat(self):
        raise NotImplementedError

    @property
    def lon(self):
        raise NotImplementedError

    @property
    def coords(self):
        raise NotImplementedError

    @property
    def gsd(self):
        raise NotImplementedError


class WebTile(Tile):

    crs = 'epsg:3857'
    proj = Proj(crs)

    @property
    def lat(self):
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * self.y / (2**self.z))))
        lat_deg = math.degrees(lat_rad)
        return lat_deg

    @property
    def lon(self):
        return self.x / (2 ** self.z) * 360. - 180.

    @property
    def coords(self):
        x, y = self.proj(self.lon, self.lat)
        return x, y

    @property
    def gsd(self):
        return 156543.03 / 2 ** self.z


class WorldTile(Tile):

    crs = 'epsg:3395'
    proj = Proj('epsg:3395')

    @property
    def lat(self):
        c =  WorldConverter(self.z, self.size)
        f = lambda x: c.lat_to_ypx(x) / self.size - self.y
        res = fsolve(f, 0, xtol=1e-11, maxfev=1000)[0]
        return res

    @property
    def lon(self):
        return self.x / (2 ** self.z) * 360. - 180.

    @property
    def coords(self):
        x, y = self.proj(self.lon, self.lat)
        return x, y

    @property
    def gsd(self):
        return 156543.03 / 2 ** self.z
