import rasterio
import numpy as np


def generate_array(width, height, count, dtype, **kwargs):
    return np.ones(shape=(count, height, width), dtype=dtype)


def create_tiff_file(filename, width, height, **kwargs):
    profile = {'width': width,
                'height': height,
                'dtype': 'uint8',
                'count': 3,
                'driver': 'GTiff',
                'transform': rasterio.Affine(0.5, 0, 100000, 0, -0.5, 20000),
                'crs': 'EPSG:3857',
                'nodata': 0}
    # we can add\modify any other creation option
    profile.update(**kwargs)
    data = generate_array(width, height, profile['count'], profile['dtype'])
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data)



