from urban.base import compose, parser
from loguru import logger
import os
from tests.testutils.generate_test_files import SYNTHETIC_DATA_PATH, RASTER_INPUT_FILENAME, RAST_EXT, VEC_EXT,\
    generate_parcels, _rasterize, generate_meta, DEFAULT_CRS, AOI
from typing import Final, Sequence
from gpdadapter import FeatureCollection
import rasterio
import numpy as np
from tests.testutils.defaults import MOCK_PIPELINES_PATH

PATH: Final[str] = os.path.join(SYNTHETIC_DATA_PATH, 'parcels')
PIPELINE_CONFIG_PATH: Final[str] = os.path.join(MOCK_PIPELINES_PATH, 'nspd_parcels_mock.yml')
REQUIRED_INPUT_FILES: Sequence[str] = (RASTER_INPUT_FILENAME+RAST_EXT, )
PARCELS_RES: Final[float] = 0.15

def test_parcels_pipeline():
    # creating test setup
    logger.info('CREATING TEST SETUP...')

    if not os.path.exists(PATH):
        logger.info('data dir not found, creating...')
        os.mkdir(PATH)

    if not all(os.path.exists(os.path.join(PATH, f)) for f in REQUIRED_INPUT_FILES):
        logger.info('Not all input files found, creating synthetic data...')
        parcels = generate_parcels()
        aoi = generate_meta([AOI(0, 0, 500, 500)]).geometry[0]
        # this test requires unique rasterization
        xmin, ymin, xmax, ymax = aoi.bounds
        width = int(np.ceil((xmax - xmin) / PARCELS_RES))
        height = int(np.ceil((ymax - ymin) / PARCELS_RES))
        transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)
        mask_R = _rasterize(parcels[:2], height, width, transform)*255
        mask_G = _rasterize(parcels[2:], height, width, transform)*255
        mask_B = np.zeros_like(mask_R)

        with rasterio.open(os.path.join(PATH, RASTER_INPUT_FILENAME+RAST_EXT), 'w', driver='GTiff',
                           height=height, width=width, count=3, dtype='uint8',
                           crs=DEFAULT_CRS, transform=transform) as dst:
            dst.write(mask_R, 1)
            dst.write(mask_G, 2)
            dst.write(mask_B, 3)



    logger.info('TEST SETUP CREATED!')

    # running pipeline
    logger.info('RUNNING PIPELINE...')
    d = parser.parse_config(PIPELINE_CONFIG_PATH)
    d = compose.Compose.from_config(d['config'])
    d(PATH)
    logger.info('PIPELINE OK!')

    # validating result
    logger.info('VALIDATING RESULTS...')
    assert len(FeatureCollection.from_file(os.path.join(PATH, 'output.geojson'))) == 2
