import json
from loguru import logger
from model import WorkflowStatus
from config import Config


def parse_we_page_response(content):
    result = {}
    try:
        workflows_data = json.loads(content)
    except Exception as e:
        logger.exception("Could not json decode responlse from workflow engine")
        return result
    for wf in workflows_data:
        try:
            result.update({wf['id']: WorkflowStatus.from_we_status(wf['status'])})
        except KeyError as e:
            logger.warning(f"Bad response from workflow engine! Expected key {e} not found in workflow response {wf}")
        except Exception as e:
            logger.warning(f"Bad response from workflow engine! Error: {e}")
    return result


def parse_we_add_workflow_response(content):
    """
    returns (worflow_ID, WorkflowStatus, AOI_ID)
    in case of errors,(None, None, None) is returned
    """
    result = None, None, None
    try:
        wf = json.loads(content)
    except Exception as e:
        logger.exception("Could not json decode responlse from workflow engine")
        return result
    try:
        result = wf['id'], WorkflowStatus.from_we_status(wf['status']), wf['areasOfInterest'][0]['id']
    except KeyError as e:
        logger.warning(f"Bad response from workflow engine! Expected key {e} not found in workflow response {wf}")
    except Exception as e:
        logger.warning(f"Bad response from workflow engine! Error: {e}")
    return result


def generate_add_workflow_request(workflow_def_id, image_id, raw_raster_uri, raster_layer_uri, aoi):
    body = {"areasOfInterest": [{"geometry": aoi}],
            "system": Config.SYSTEM_ID,
            "params": {
                "priority": 9,
                "source_type": "local",
                "url": raw_raster_uri,
                "raster-layer-uri": raster_layer_uri
            },
            "processingId": str(image_id),
            "artifacts": [{
                "artifactType": "RAW_RASTER",
                "uri": raw_raster_uri
            }],
            "workflowDefinitionId": workflow_def_id}
    return body
