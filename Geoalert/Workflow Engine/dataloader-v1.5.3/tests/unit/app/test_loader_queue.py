import pytest
import json
from app.config import config
from app.loader.queue import MessageHandler, lower_source_type_input_kwargs

message_example = {'task_id': 3814483, 
                   'input': {'source_type': 'xYz', 
                             'zoom': '16', 
                             'geometry': {'type': 'Polygon', 
                                          'coordinates': [[[ 13.348043400129663, 52.515478407888835 ], 
                                                           [ 13.348729807195298, 52.513376471325707 ], 
                                                           [ 13.352497743853467, 52.513452018356347 ], 
                                                           [ 13.351650688325664, 52.51596277225017 ], 
                                                           [ 13.348043400129663, 52.515478407888835 ]]]}, 
                             'projection': 'epsg:3857', 
                             'workers': '2', 
                             'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'}, 
                   'output': {'bucket': 'workflow-white-maps', 'filename': 'workflow-tmp/9debbcf9-6fec-47fe-a10e-7e367455df75/area-3814471.tif'}, 
                   'link': 'http://engine.workflow-staging.svc:8080/api/v0/tasks/3814483/runcheck'}


def test_create_loader_kwargs():
    handler = MessageHandler(connection=None, input_queue=None, output_queue=None, storage=None, config=config)
    body = json.dumps(message_example)
    input_kwargs = json.loads(body)
    input_kwargs = lower_source_type_input_kwargs(input_kwargs)
    _, loader_kwargs = handler.create_loader_setup(input_kwargs)
    source_type = loader_kwargs['source_type']
    assert source_type == 'xyz'
