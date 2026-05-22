import os
import rasterio

from app.config import config
from app.loader.queue import MessageHandler, lower_source_type_input_kwargs

message_example = {'task_id': 3814483,
                   'input': {'source_type': 'xYz',
                             'zoom': '16',
                             'geometry': {'type': 'Polygon',
                                          'coordinates': [[[ 47.96775967074754, 42.416168999761126 ],
                                                           [ 47.968221404489057, 42.416229225901326 ],
                                                           [ 47.969225173492347, 42.412063584537655 ],
                                                           [ 47.968642987470439, 42.411943132257257 ],
                                                           [ 47.96775967074754, 42.416168999761126 ] ]]},
                             'projection': 'epsg:3857',
                             'workers': '2',
                             'url': 'https://rasters-staging.mapflow.ai/api/v0/cogs/tiles/{z}/{x}/{y}.png?uri=s3://healthcheck/area-116229.tif',
                             'header': {'x-api-key':'mySuperSecretKey'}
                             },
                   'output': {'bucket': 'workflow-white-maps', 'filename': 'workflow-tmp/9debbcf9-6fec-47fe-a10e-7e367455df75/area-3814471.tif'},
                   'link': 'http://engine.workflow-staging.svc:8080/api/v0/tasks/3814483/runcheck'}


def test_create_loader_kwargs():
    handler = MessageHandler(connection=None, input_queue=None, output_queue=None, storage=None, config=config)
    loader_kwargs = lower_source_type_input_kwargs(message_example)
    loader, loader_kwargs = handler.create_loader_setup(loader_kwargs)
    assert loader_kwargs['header'] == {'x-api-key':'mySuperSecretKey'}
    handler.download_raster(loader, **loader_kwargs)
    out_filename = loader_kwargs['output_fp']
    assert os.path.exists(out_filename) and os.path.isfile(out_filename)
    # check that the image is downloaded and the result is non-zero
    with rasterio.open(out_filename) as src:
        assert src.read().max() > 0