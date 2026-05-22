# THIS IS AN EXAMPLE, DO NOT MODIFY THIS FILE
# because it prevents from auto-merging branches when every branch has differences in it
# copy it and add the copy to .gitignore

import sys
from modules.urban.urban.base import compose, parser
from loguru import logger
#import cProfile as profile  # for profiling
logger.add(sys.stdout, level="TRACE")


if __name__ == '__main__':
    #pr = profile.Profile()
    #pr.disable()

    d = parser.parse_config('tests/test_pipelines/manual_tests/buildings_default.yml') #path to config

    # simple (non-blocked) config
    #d = compose.Compose.from_config(d['config'])

    # buildings with heights
    # d = compose.Compose.from_config(d['config'], {'Segmentation': True,
    #                                               'Simplification': False,
    #                                               'ShadowsWallsHeights': True,
    #                                               'HeightRegression': True,
    #                                               'HeightsFromEmbedding': True,
    #                                               'HeightsByArea': True,
    #                                               'GenerateFootprints': True,
    #                                               'Add_Roofs&Shadows&Walls': False})

    # buildings default
    d = compose.Compose.from_config(d['config'], {'Postprocessing1': True,
                                                 'Simplification': True,
                                                 'OSM': False,
                                                 'AlignBuildings': True,
                                                 'Classification': False,
                                                 'Postprocessing2': True})

    #pr.enable()
    d('tests/test_data/real/Ufa')  # path to your data here
    #pr.disable()
    #pr.dump_stats('profile_new.pstat')
