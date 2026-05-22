import math
from rasterio import transform, warp, crs, Affine

WEB_MERCATOR = crs.CRS.from_epsg(3857)
WEB_MERCATOR_BOUND = 85.05
CRS_LATLON = crs.CRS.from_epsg(4326)
DEFAULT_IMAGE_SIZE = 1000


def get_pixel_size(src_transform, src_crs, width=DEFAULT_IMAGE_SIZE, height=DEFAULT_IMAGE_SIZE):
    """
    gets the GSD (ground sample distance) - real linear unit whic corresponds to pixel size.
    We cannot use the internal representation in transform, because even if it is in meters,
    it can be warped and change depending on location

    So we reproject to web mercator and calculate the pixel size there, with known (cos(lat)) factor

    Default image size allows skip the image width and height arguments, but for better results they should be specified
    to see the GSD for the center of image, not a point at certain distance from corner
    """
    if not isinstance(src_transform, Affine):
        src_transform = Affine(*src_transform)
    bounds = transform.array_bounds(height, width, src_transform)
    image_center = ([(bounds[0] + bounds[2])/2], [(bounds[1] + bounds[3])/2])
    center_latitude = warp.transform(src_crs, crs.CRS.from_epsg(4326), *image_center)[1][0]
    if abs(center_latitude) > WEB_MERCATOR_BOUND:
        raise ValueError(f'Image with {bounds = } and {src_crs = } lies at {center_latitude = }, out of latitude range '
                         f'[-{WEB_MERCATOR_BOUND}, {WEB_MERCATOR_BOUND}] allowed for web mercator')
    # we reproject to web mercator as it is (almost) universal and measures in meters, and easy to convert to 'real' gsd
    dst_crs = WEB_MERCATOR
    # we find the appropriate transfrom in the CRS that does not change scale
    new_transform = warp.calculate_default_transform(src_crs, dst_crs, width, height, *bounds)[0]
    res_merc = (abs(new_transform.a) + abs(new_transform.e)) / 2  # mean between xres and yres
    # mercator 'resolution' is measured in meters at equator, but the real pixel size changes as cos(latitude)
    # so we have to calculate it
    res = res_merc*math.cos(center_latitude*math.pi/180)
    return res
