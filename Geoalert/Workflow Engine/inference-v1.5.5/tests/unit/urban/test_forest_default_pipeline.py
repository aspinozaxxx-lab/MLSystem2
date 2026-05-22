from urban.base import compose, parser
from loguru import logger
import os
from tests.testutils.generate_test_files import SYNTHETIC_DATA_PATH, RASTER_INPUT_FILENAME, RAST_EXT, VEC_EXT,\
    generate_forest, generate_raster, generate_meta, SEMANTIC_FILENAMES
from tests.testutils.validate import validate_collection
from typing import Final
from gpdadapter import FeatureCollection
from tests.testutils.defaults import SemanticClass, MOCK_PIPELINES_PATH


PATH: Final[str] = os.path.join(SYNTHETIC_DATA_PATH, 'forest_default')
PIPELINE_CONFIG_PATH: Final[str] = os.path.join(MOCK_PIPELINES_PATH, 'forest_default_mock.yml')
PIPELINE_BLOCKS_OPTIONS = {'Segmentation': True,
                           'Heights': True,
                           'Postprocessing': True}
REQUIRED_INPUT_FILES = (RASTER_INPUT_FILENAME+RAST_EXT, )


def test_buildings_default_pipeline():
    # creating test setup
    logger.info('CREATING TEST SETUP...')

    if not os.path.exists(PATH):
        logger.info('data dir not found, creating...')
        os.mkdir(PATH)

    if not all(os.path.exists(os.path.join(PATH, f)) for f in REQUIRED_INPUT_FILES):
        logger.info('Not all input files found, creating synthetic data...')
        aoi = generate_meta(with_angles=False)[0, 'geometry']  # AOI Polygon in 3857
        forest = generate_forest()
        forest.to_file(os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.FOREST]+VEC_EXT))
        generate_raster(os.path.join(PATH, RASTER_INPUT_FILENAME+RAST_EXT),
                        aoi=aoi,
                        vectors={SemanticClass.FOREST: forest})

    logger.info('TEST SETUP CREATED!')

    # running pipeline
    logger.info('RUNNING PIPELINE...')
    d = parser.parse_config(PIPELINE_CONFIG_PATH)
    d = compose.Compose.from_config(d['config'], PIPELINE_BLOCKS_OPTIONS)
    d(PATH)
    logger.info('PIPELINE OK!')

    # validating result
    # logger.info('VALIDATING RESULTS...')
    # assert validate_collection(FeatureCollection.from_file(os.path.join(PATH, 'output.geojson')),
    #                            FeatureCollection.from_file(
    #                                os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.FOREST]+VEC_EXT)))

