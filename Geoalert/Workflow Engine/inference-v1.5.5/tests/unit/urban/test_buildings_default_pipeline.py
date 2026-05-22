from urban.base import compose, parser
from loguru import logger
import os
from tests.testutils.generate_test_files import SYNTHETIC_DATA_PATH, RASTER_INPUT_FILENAME, RAST_EXT, VEC_EXT,\
    generate_osm_buildings, generate_osm_roads, generate_rooftops, generate_raster, generate_meta, \
    OSM_BUILDINGS_NAME, OSM_ROADS_NAME, generate_cnt, SEMANTIC_FILENAMES
from tests.testutils.validate import validate_collection
from typing import Final
from gpdadapter import FeatureCollection
from tests.testutils.defaults import SemanticClass, MOCK_PIPELINES_PATH


PATH: Final[str] = os.path.join(SYNTHETIC_DATA_PATH, 'buildings_default')
PIPELINE_CONFIG_PATH: Final[str] = os.path.join(MOCK_PIPELINES_PATH, 'buildings_default_mock.yml')
PIPELINE_BLOCKS_OPTIONS = {'Segmentation': True,
                           'Postprocessing1': True,
                           'Simplification': True,
                           'OSM': True,
                           'Classification': True,
                           'Postprocessing2': True}
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
        rt = generate_rooftops()
        cnt = generate_cnt()
        rt.to_file(os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.ROOFTOP]+VEC_EXT))
        generate_osm_buildings().to_file(os.path.join(PATH, OSM_BUILDINGS_NAME+VEC_EXT))
        generate_osm_roads().to_file(os.path.join(PATH, OSM_ROADS_NAME+VEC_EXT))
        generate_raster(os.path.join(PATH, RASTER_INPUT_FILENAME+RAST_EXT),
                        aoi=aoi,
                        vectors={SemanticClass.ROOFTOP: rt,
                                 SemanticClass.BLD_CONTOUR: cnt})

    logger.info('TEST SETUP CREATED!')

    # running pipeline
    logger.info('RUNNING PIPELINE...')
    d = parser.parse_config(PIPELINE_CONFIG_PATH)
    d = compose.Compose.from_config(d['config'], PIPELINE_BLOCKS_OPTIONS)
    d(PATH)
    logger.info('PIPELINE OK!')

    # validating result
    logger.info('VALIDATING RESULTS...')
    assert validate_collection(FeatureCollection.from_file(os.path.join(PATH, 'output.geojson')),
                               FeatureCollection.from_file(
                                   os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.ROOFTOP]+VEC_EXT)),
                               categorical_properties=('shape_type', 'class_id'))

