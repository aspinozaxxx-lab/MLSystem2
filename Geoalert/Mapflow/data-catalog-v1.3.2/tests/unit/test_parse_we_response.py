import json
import aiohttp
import asyncio
from model.workflow import WorkflowStatus
from tests.mock.sample_we_response_body import page_response, add_workflow_response
from service.workflow import request_batch, start_workflow
from aioresponses import aioresponses
from app.functional.workflow_engine import (parse_we_add_workflow_response,
                                        parse_we_page_response,
                                        generate_add_workflow_request)


mock_url_page = 'https://workflow.engine.url/rest/v0/workflows/page'


async def _request_batch():
    semaphore = asyncio.Semaphore(1)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        result = await request_batch(url=mock_url_page,
                                     auth=None,
                                     workflows_batch=[4410719, 4410901],
                                     session=session,
                                     semaphore=semaphore)
    return result


def test_request_batch():
    with aioresponses() as mocked:
        response_body = page_response
        mocked.post(mock_url_page, status=200, body=response_body)
        result = asyncio.run(_request_batch())
        assert result == {4410719: WorkflowStatus.OK, 4410901: WorkflowStatus.OK}


def test_parse_we_page_response():
    assert parse_we_page_response(page_response) == {4410719: WorkflowStatus.OK, 4410901: WorkflowStatus.OK}


def test_parse_add_workflow_response():
    assert parse_we_add_workflow_response(add_workflow_response) == (4412998, WorkflowStatus.IN_PROGRESS, 4412999)


def test_generate_add_workflow_request():
    result = generate_add_workflow_request(workflow_def_id=4409285,
                                         image_id="MyAwesomeProcessing",
                                         raw_raster_uri="s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/f184b344-dcce-4710-9cb1-0fda4dc5bb8f/0f0353ff978d4998bdc994704ceeace8.tif",
                                         raster_layer_uri="s3://test-bucket/3/0f0353ff978",
                                         aoi={"type": "Polygon",
                                              "coordinates": [ [ [ 47.96080936452077, 42.409955669630733 ],
                                                                 [ 47.976346120449676, 42.409955669630733 ],
                                                                 [ 47.976346120449676, 42.417112542624224 ],
                                                                 [ 47.96080936452077, 42.417112542624224 ],
                                                                 [ 47.96080936452077, 42.409955669630733 ] ] ] }
                                         )
    expected = json.loads("""{
    "areasOfInterest": [
    {
      "geometry": { "type": "Polygon", "coordinates": [ [ [ 47.96080936452077, 42.409955669630733 ], [ 47.976346120449676, 42.409955669630733 ], [ 47.976346120449676, 42.417112542624224 ], [ 47.96080936452077, 42.417112542624224 ], [ 47.96080936452077, 42.409955669630733 ] ] ] }
    }
    ],
    "system":"data-catalog",
    "params":{
      "priority":9,
      "source_type":"local",
      "url": "s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/f184b344-dcce-4710-9cb1-0fda4dc5bb8f/0f0353ff978d4998bdc994704ceeace8.tif",
      "raster-layer-uri": "s3://test-bucket/3/0f0353ff978"
    },
    "processingId": "MyAwesomeProcessing",
    "artifacts": [{
        "artifactType": "RAW_RASTER",
        "uri": "s3://users-data/a.trekin@geoalert.io_1a761087-20a6-4436-b202-15ac857d27ea/f184b344-dcce-4710-9cb1-0fda4dc5bb8f/0f0353ff978d4998bdc994704ceeace8.tif"
    }],
    "workflowDefinitionId": 4409285
    }""")

    assert result == expected
