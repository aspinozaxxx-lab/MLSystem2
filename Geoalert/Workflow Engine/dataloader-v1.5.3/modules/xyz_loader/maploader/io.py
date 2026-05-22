import rasterio
import numpy as np


def save_image(path, image, **kwargs):

    h, w, c = image.shape

    with rasterio.open(path, 'w', height=h, width=w, count=c,
                       driver='GTiff', nodata=0, dtype=image.dtype, **kwargs) as dst:

        clipped_image = np.clip(image, 1, np.max(image))
        dst.write(clipped_image.transpose(2, 0, 1))


def read_image(path):
    with rasterio.open(path) as src:
        return src.read()
