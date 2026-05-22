import rasterio
import numpy as np


def generate_array(width, height, count, dtype, value=1, matrix='ones'):
    if matrix == 'ones':
        array = np.ones(shape=(count, height, width), dtype=dtype) * value
    elif matrix == 'zeros':
        array = np.zeros(shape=(count, height, width), dtype=dtype)
    elif matrix == 'eye':
        array = np.eye(height, width, dtype=dtype) * value
        array = np.stack([array] * count, axis=0)
    else:
        raise ValueError(f"Unknown matrix: {matrix}")

    return array


def create_tiff_file(filename, value=1, matrix='ones', data=None, **kwargs):
    profile = {'dtype': 'uint8',
               'count': 3,
               'driver': 'GTiff',
               'transform': rasterio.Affine(10.0, 0, 100000, 0, -10.0, 20000),
               'crs': 'EPSG:3857',
               'nodata': 0}
    # we can add\modify any other creation option
    profile.update(**kwargs)
    if data is None:
        data = generate_array(profile['width'], profile['height'], profile['count'], profile['dtype'],
                              value=value, matrix=matrix)

    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(data)

    print(f"Created: {filename}")
