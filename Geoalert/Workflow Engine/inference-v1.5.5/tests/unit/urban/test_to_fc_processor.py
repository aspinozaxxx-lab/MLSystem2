from urban.bricks.model_bricks.nspd_parcels import ToFeatureCollectionProcessor
from aeronet_raster import BandCollection
import numpy as np
from rasterio.io import MemoryFile
from typing import Tuple, List

def processing_fn(x: np.array) -> Tuple[List[np.array], List[float]]:
    """Mocks NSPD parcels model"""
    assert x.ndim == 3
    if np.any(x > 0):
        return [x[0]], [1]
    else:
        return [x[0]], [0]


def test_to_fc_processor():
    """creates a single band with 4 squares"""
    width, height = 256, 256
    polygon_centers_coords = ((64, 64), (192, 192), (64, 192), (192, 64))
    polygon_halfsize = 32

    data = np.zeros((height, width), np.uint8)
    for y, x in polygon_centers_coords:
        data[y-polygon_halfsize:y+polygon_halfsize, x-polygon_halfsize:x+polygon_halfsize] = 1

    with MemoryFile() as memfile:
        with memfile.open(driver='GTiff', count=1, width=width, height=height, dtype=np.uint8,
                          crs='EPSG:3857') as d:
            d.write(data, 1)

        bc = BandCollection([memfile, ])
        processor = ToFeatureCollectionProcessor([0], processing_fn, (128, 128), 0)
        fc = processor.process(bc)

    assert len(fc) == 4 # all 4 polygons found

