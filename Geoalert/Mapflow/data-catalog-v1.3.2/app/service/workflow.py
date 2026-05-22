import json
import asyncio
import aiohttp
from typing import List, Optional
from loguru import logger
from shapely.geometry import mapping
from sqlalchemy.exc import DBAPIError

from crud import (get_workflows_by_status,
                  update_statuses,
                  get_image_by_id,
                  get_workflow_def,
                  update_workflow_by_image_id,
                  get_mosaic_by_id)
from model import Workflow, WorkflowStatus, Data
from functional.workflow_engine import parse_we_add_workflow_response, parse_we_page_response, generate_add_workflow_request
from service.data import add_coglink_to_images_table_service


from config import Config


async def request_all_batches(url: str,
                              auth: Optional[aiohttp.BasicAuth],
                              workflows: list,
                              batch_size: int,
                              session: aiohttp.ClientSession,
                              semaphore: asyncio.Semaphore):
    all_keys = [wf.we_id for wf in workflows]
    key_batches = [all_keys[batch_size*i: batch_size*(i+1)] for i in range(len(all_keys)//batch_size + 1)]
    all_res = {}
    for batch in key_batches:
        batch_result = await request_batch(url=url,
                                           auth=auth,
                                           workflows_batch=batch,
                                           session=session,
                                           semaphore=semaphore)
        all_res.update(batch_result)
    return all_res


async def request_batch(url: str,
                        auth: Optional[aiohttp.BasicAuth],
                        workflows_batch: List[str],
                        session: aiohttp.ClientSession,
                        semaphore: asyncio.Semaphore):
    body = {"filter": {"workflowIds": workflows_batch}}
    status = None
    try:
        async with semaphore:
            status, content = await send_request(url=url, body=body, session=session, auth=auth)
    except Exception as e:
        logger.exception(f'Error in wf request! Status {status}')
        return {}
    else:
        return parse_we_page_response(content)


async def send_request(url, body, session, auth):

    async with session.post(url,
                            data=json.dumps(body),
                            headers={'Content-Type': 'application/json'},
                            auth=auth,
                            timeout=10,
                            raise_for_status=True) as response:
        status = response.status
        content = await response.read()
    return status, content


async def start_workflow(url: str,
                         auth: Optional[aiohttp.BasicAuth],
                         workflow_def_id: int,
                         image: Data,
                         session: aiohttp.ClientSession,
                         semaphore: asyncio.Semaphore):
    mosaic = get_mosaic_by_id(image.mosaic_id)
    image.footprint = mapping(image.footprint)
    body = generate_add_workflow_request(workflow_def_id,
                                         image_id=image.id,
                                         raw_raster_uri=image.image_url,
                                         raster_layer_uri=mosaic.cog_link,
                                         aoi=image.footprint)
    status = None
    try:
        async with semaphore:
            status, content = await send_request(url=url, body=body, session=session, auth=auth)
    except aiohttp.client_exceptions.ClientResponseError as e:
        if 400 == e.status:
            # We treat client error as FAIL of processing
            return None, WorkflowStatus.FAILED, None
        else:
            return None, None, None
    except Exception as e:
        logger.exception(f'Error in wf request! Status {status}')
        return None, None, None
    else:
        return parse_we_add_workflow_response(content)


async def poll():
    semaphore = asyncio.Semaphore(1)
    logger.info("Start polling Workflow Engine")
    if Config.WORKFLOW_ENGINE_LOGIN or Config.WORKFLOW_ENGINE_PASSWORD:
         = aiohttp.BasicAuth(Config.WORKFLOW_ENGINE_LOGIN, Config.WORKFLOW_ENGINE_PASSWORD)
    else:
        auth = None
    page_url = Config.WORKFLOW_ENGINE_URL + '/api/v0/workflows/page'
    start_url = Config.WORKFLOW_ENGINE_URL + '/api/v0/workflows'
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1)) as session:
        while True:
            try:
                logger.debug(f"Sleeping for {Config.SLEEP}")
                await asyncio.sleep(Config.SLEEP)
                # Check status of successfully sent workflows
                workflows = get_workflows_by_status([WorkflowStatus.IN_PROGRESS.value])
                if workflows:
                    logger.debug(f"Sending request to WorkflowEngine for IDs: {[wf.we_id for wf in workflows]}")
                    updated_workflows = await request_all_batches(url=page_url,
                                                                  auth=auth,
                                                                  workflows=workflows,
                                                                  batch_size=Config.WORKFLOW_ENGINE_BATCH_SIZE,
                                                                  session=session,
                                                                  semaphore=semaphore)
                    logger.debug(f"Got {updated_workflows} in response")
                    update_statuses(updated_workflows)
                else:
                    logger.debug("Nothing to update")

                workflows = get_workflows_by_status([WorkflowStatus.UNPROCESSED.value])
                workflow_def_id = get_workflow_def().id
                if workflows:
                    logger.debug(f"Starting new workflows for IDs: {[wf.image_id for wf in workflows]}")
                    for workflow in workflows:
                        workflow_id, status, aoi_id = await start_workflow(url=start_url,
                                                                           auth=auth,
                                                                           workflow_def_id=workflow_def_id,
                                                                           image=get_image_by_id(workflow.image_id),
                                                                           session=session,
                                                                           semaphore=semaphore)
                        if workflow_id:
                            logger.debug(f"Successfully started workflow, got id {workflow_id}, status {status}")
                            update_workflow_by_image_id(workflow.image_id, workflow_id, status)
                            add_coglink_to_images_table_service(image_id=workflow.image_id, aoi_id=aoi_id)
                        elif status == WorkflowStatus.FAILED:
                            # processing could not start and is immadiately FAILED
                            logger.warning(f"Could not start workflow for image {workflow.image_id}, marking it as FAILED")
                            update_workflow_by_image_id(workflow.image_id, None, status)
                else:
                    logger.debug("Nothing to start")
            except DBAPIError:
                logger.warning(f"Database is not available. Sleeping for {Config.SLEEP} seconds")
                await asyncio.sleep(Config.SLEEP)
                continue


def start_polling_we():
    asyncio.run(poll())
