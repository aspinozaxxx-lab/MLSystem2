from urban.base import compose, parser
from loguru import logger
import os
from tests.testutils.generate_test_files import SYNTHETIC_DATA_PATH, RASTER_INPUT_FILENAME, RAST_EXT, VEC_EXT, \
    generate_roads, generate_raster, generate_meta, ROADS_NAME
from typing import Final
from tests.testutils.defaults import SemanticClass, MOCK_PIPELINES_PATH


PATH: Final[str] = os.path.join(SYNTHETIC_DATA_PATH, 'roads_default')
PIPELINE_CONFIG_PATH: Final[str] = os.path.join(MOCK_PIPELINES_PATH, 'roads_default_mock.yml')
REQUIRED_INPUT_FILES = (RASTER_INPUT_FILENAME+RAST_EXT, )


def test_roads_default_pipeline():
    # creating test setup
    logger.info('CREATING TEST SETUP...')

    if not os.path.exists(PATH):
        logger.info('data dir not found, creating...')
        os.mkdir(PATH)

    if not all(os.path.exists(os.path.join(PATH, f)) for f in REQUIRED_INPUT_FILES):
        logger.info('Not all input files found, creating synthetic data...')
        aoi = generate_meta(with_angles=False)[0, 'geometry']  # AOI Polygon in 3857
        roads = generate_roads()
        roads.to_file(os.path.join(PATH, ROADS_NAME+VEC_EXT))
        generate_raster(os.path.join(PATH, RASTER_INPUT_FILENAME+RAST_EXT),
                        aoi=aoi,
                        vectors={SemanticClass.ROAD: roads})

    logger.info('TEST SETUP CREATED!')

    # running pipeline
    logger.info('RUNNING PIPELINE...')
    d = parser.parse_config(PIPELINE_CONFIG_PATH)
    d = compose.Compose.from_config(d['config'])
    d(PATH)
    logger.info('PIPELINE OK!')

    # validating result
    pass
