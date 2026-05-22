from shapely.geometry import Polygon
from inference.message import InferenceInputMessage


def test_input_message():
    raw_input_message = {'task_id': '306712',
                         'runcheck_url': 'http://engine.workflow-duty.svc:8080/api/v0/tasks/306712/runcheck',
                         'input': {'aoi':
                                        {'type': 'Polygon',
                                         'coordinates': [[[47.96746730798089, 42.41012607136867],
                                                         [47.96746730798089, 42.41694214088627],
                                                         [47.9696881769895, 42.41694214088627],
                                                         [47.9696881769895, 42.41012607136867],
                                                         [47.96746730798089, 42.41012607136867]]]},
                                   'pipeline': 'SOME_PIPELINE==',
                                   'source_data': [{'path': 's3://users-data/healthcheck.tif',
                                                    'name': 'input.tif'}]
                                   },
                         'output': {
                            'output_data': [{'path': 's3://users-data/output.geojson',
                                            'name': 'output.geojson'}]}
    }
    message = InferenceInputMessage(**raw_input_message)

    assert isinstance(message.input.aoi, Polygon)
