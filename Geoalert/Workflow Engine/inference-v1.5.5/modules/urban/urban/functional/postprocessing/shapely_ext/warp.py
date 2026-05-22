from shapely.geometry import mapping, shape
from rasterio.warp import transform_geom
from aeronet_raster.utils.coords import _utm_zone

CRS_LATLON = 'EPSG:4326'


def reproject_to_utm(object):
    """
    Reprojects the shape from lat-lon to utm
    """
    if not object:
        return object
    centroid = object.centroid
    crs = _utm_zone(centroid.y, centroid.x)
    geometry = transform_geom(src_crs=CRS_LATLON,
                              dst_crs=crs,
                              geom=mapping(object))
    return shape(geometry)
