import math
from .errors import CrsIsNotSupported


def get_converter(projection):

    converters = {
        'epsg:3857': WebConverter,
        'epsg:3395': WorldConverter,
    }
    projection = projection.lower()

    if projection in converters.keys():
        c = converters[projection]
    else:
        raise CrsIsNotSupported(list(converters.keys()), projection)

    return c


class Converter:

    def __init__(self, z, tile_size):
        self.z = z
        self.tile_size = tile_size

    def lat_to_ypx(self, lat):
        raise NotImplementedError

    def lon_to_xpx(self, lon):
        raise NotImplementedError

    def lat_to_ytile(self, lat):
        raise NotImplementedError

    def lon_to_xtile(self, lon):
        raise NotImplementedError


class WebConverter(Converter):

    def lat_to_ypx(self, lat):
        lat = math.radians(lat)
        t = math.log(((1 + math.sin(lat)) / (1 - math.sin(lat)))) / math.pi
        y = (1 - 0.5*t) * self.tile_size * 2 ** (self.z - 1)
        return y

    def lon_to_xpx(self, lon):
        x = (2 ** self.z * self.tile_size) * (lon + 180) / 360
        return x

    def lat_to_ytile(self, lat):
        return int(self.lat_to_ypx(lat) // self.tile_size)

    def lon_to_xtile(self, lon):
        return int(self.lon_to_xpx(lon) // self.tile_size)


class WorldConverter(Converter):

    e = 0.08181919092890638

    def lat_to_ypx(self, lat):

        lat = math.radians(lat)
        corr = ((1 - self.e * math.sin(lat)) / (1 + self.e * math.sin(lat))) ** self.e
        t = math.log(((1 + math.sin(lat))/ (1 - math.sin(lat))) * corr) / math.pi
        y = (1 - 0.5 * t) * self.tile_size * 2 ** (self.z - 1)

        return y

    def lon_to_xpx(self, lon):
        x = (2 ** self.z * self.tile_size) * (lon + 180) / 360
        return x

    def lat_to_ytile(self, lat):
        return int(self.lat_to_ypx(lat) // self.tile_size)

    def lon_to_xtile(self, lon):
        return int(self.lon_to_xpx(lon) // self.tile_size)
