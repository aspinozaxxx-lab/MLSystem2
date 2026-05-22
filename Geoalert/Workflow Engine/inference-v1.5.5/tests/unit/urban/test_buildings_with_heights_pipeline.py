from urban.base import compose, parser
from urban.base.defaults import AOI_FILENAME
from loguru import logger
import os
from tests.testutils.generate_test_files import SYNTHETIC_DATA_PATH, RASTER_INPUT_FILENAME, RAST_EXT, VEC_EXT, \
    generate_osm_buildings, generate_osm_roads, generate_rooftops, generate_raster, generate_meta, \
    OSM_BUILDINGS_NAME, OSM_ROADS_NAME, SEMANTIC_FILENAMES, generate_cnt, WLL_NAME, SHL_NAME, generate_footprints,\
    generate_shadows, generate_walls, generate_zkh, generate_sh_markup, generate_wl_markup, META_NAME, ZKH_NAME
from typing import Final
from tests.testutils.defaults import SemanticClass, MOCK_PIPELINES_PATH
from gpdadapter import FeatureCollection
from tests.testutils.validate import validate_collection

PATH: Final[str] = os.path.join(SYNTHETIC_DATA_PATH, 'buildings_with_heights')
PIPELINE_CONFIG_PATH: Final[str] = os.path.join(MOCK_PIPELINES_PATH, 'buildings_with_heights_mock.yml')
PIPELINE_BLOCKS_OPTIONS = {'Segmentation': True,
                           'Postprocessing1': True,
                           'Heights': True,
                           'Simplification': True,
                           'OSM': True,
                           'Classification': True,
                           'ZKH': True,
                           'Postprocessing2': True}
REQUIRED_INPUT_FILES = (RASTER_INPUT_FILENAME+RAST_EXT, META_NAME+VEC_EXT, WLL_NAME+VEC_EXT, SHL_NAME+VEC_EXT)


def test_buildings_with_heights_pipeline():
    # creating test setup
    logger.info('CREATING TEST SETUP...')

    if not os.path.exists(PATH):
        logger.info('data dir not found, creating...')
        os.mkdir(PATH)

    if not all(os.path.exists(os.path.join(PATH, f)) for f in REQUIRED_INPUT_FILES):
        logger.info('Not all input files found, creating synthetic data...')
        meta = generate_meta(with_angles=False)
        aoi = meta[0, 'geometry']  # AOI Polygon in 3857
        meta.to_file(os.path.join(PATH, META_NAME + VEC_EXT))
        meta.to_file(os.path.join(PATH, AOI_FILENAME + VEC_EXT))  # also save meta as AOI
        rt = generate_rooftops()
        generate_footprints().to_file(os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.FOOTPRINT] + VEC_EXT))
        sh = generate_shadows()
        wl = generate_walls()
        cnt = generate_cnt()

        generate_wl_markup().to_file(os.path.join(PATH, WLL_NAME + VEC_EXT))
        generate_sh_markup().to_file(os.path.join(PATH, SHL_NAME + VEC_EXT))
        generate_zkh().to_file(os.path.join(PATH, ZKH_NAME + VEC_EXT))
        generate_osm_buildings().to_file(os.path.join(PATH, OSM_BUILDINGS_NAME + VEC_EXT))
        generate_osm_roads().to_file(os.path.join(PATH, OSM_ROADS_NAME + VEC_EXT))

        generate_raster(os.path.join(PATH, RASTER_INPUT_FILENAME + RAST_EXT),
                        aoi=aoi,
                        vectors={
                            SemanticClass.SHADOW: sh,
                            SemanticClass.WALL: wl,
                            SemanticClass.ROOFTOP: rt,
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
                                   os.path.join(PATH, SEMANTIC_FILENAMES[SemanticClass.FOOTPRINT]+VEC_EXT)),
                               categorical_properties=('shape_type', 'class_id'),
                               numerical_properties=(('building_height', 5),))
