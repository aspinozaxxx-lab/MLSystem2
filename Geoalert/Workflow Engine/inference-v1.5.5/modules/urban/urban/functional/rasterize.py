import numpy as np
import rasterio
from rasterio.features import geometry_mask
from aeronet_raster import BandSample
from gpdadapter import FeatureCollection
from rasterio.transform import Affine
from typing import Union


def rasterize(fc: FeatureCollection,
              transform: Affine,
              out_shape: Union[tuple, list],
              crs: rasterio.CRS,
              name='mask'):
    """Transform vector geometries to raster form, return band sample where
    raster is array of bool dtype (`True` value correspond to object area)
    Args:
        fc: `FeatureCollection` object
        transform: Affine transformation object
            Transformation from pixel coordinates of `source` to the
            coordinate system of the input `shapes`. See the `transform`
            property of dataset objects.
        out_shape: tuple or list
            Shape of output numpy ndarray.
        crs: source band crs
        name: output sample name, default `mask`
    Returns:
        `BandSample` object
    """
    if fc.empty:
        mask = np.zeros(out_shape, dtype='uint8')
    else:
        geometries = fc.to_crs(crs).geometry.tolist()
        mask = geometry_mask(geometries, out_shape=out_shape, transform=transform, invert=True).astype('uint8')

    return BandSample(name, mask, crs, transform)
