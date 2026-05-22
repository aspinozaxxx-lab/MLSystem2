from typing import Union, List, Optional
from uuid import UUID
from loguru import logger

from fastapi import APIRouter, HTTPException, status, Depends, Response, Query, File, UploadFile
from starlette.requests import Request

from schemas.data import DataReturnSchema, DataReturnErrorSchema
from schemas.mosaic import (MosaicUpdateRequestSchema, MosaicReturnSchema, MosaicReturnErrorSchema,
                            MosaicCreateResponseSchema, MosaicUpdateResponseSchema, MosaicCreateRequestSchema)
from schemas.data import LinkImageSchema
from schemas.http_credentials import HTTPCredentialsCustom
from service.mosaic import (mosaic_create_service, mosaic_update_service,
                            mosaic_retrieve_service_by_id, mosaic_delete_service, mosaic_retrieve_service,
                            delete_image_by_id_service, create_mosaic_and_file_upload_service,
                            get_image_by_id_service, get_mosaic_images_service,
                            legacy_whitemaps_upload_service, link_image_from_minio_to_mosaic_service,
                            upload_images_to_existing_mosaic_service)
from errors import DataCatalogError, MemoryLimitExceeded, FileTooBig
from dependencies.user_update_dependencies import user_memory_limit_update
from dependencies.auth_dependencies import StandardHTTPSecurity
from config import Config

router = APIRouter()

security = StandardHTTPSecurity()

ROUTE_PREFIX = Config.ROUTE_PREFIX


@router.post(f"{ROUTE_PREFIX}/mosaic", status_code=status.HTTP_200_OK, response_model=MosaicCreateResponseSchema)
def create_mosaic(request: Request, response: Response,
                  mosaic: MosaicCreateRequestSchema,
                  credentials: HTTPCredentialsCustom = Depends(security)):
    """
    Creates an empty mosaic.
    :return: mosaic
    """
    try:
        res = mosaic_create_service(name=mosaic.name, tags=mosaic.tags, username=credentials.username)
    except Exception as e:
        logger.exception(f"Database record can't be created. Exception: {str(e)}")
        raise HTTPException(500)
    return res


@router.put(ROUTE_PREFIX + "/mosaic/{mosaic_id}", status_code=status.HTTP_200_OK,
            response_model=Union[MosaicUpdateResponseSchema, MosaicReturnErrorSchema])
def update_mosaic(request: Request, response: Response,
                  mosaic_id: UUID, mosaic: MosaicUpdateRequestSchema,
                  credentials: HTTPCredentialsCustom = Depends(security)):
    try:
        res = mosaic_update_service(username=credentials.username, mosaic_id=mosaic_id, mosaic=mosaic)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic/{mosaic_id} GET
@router.get(ROUTE_PREFIX + "/mosaic/{mosaic_id}", status_code=status.HTTP_200_OK,
            response_model=Union[MosaicReturnSchema, MosaicReturnErrorSchema])
def get_mosaic_by_id(request: Request, response: Response,
                     mosaic_id: UUID,
                     credentials: HTTPCredentialsCustom = Depends(security)):
    # get mosaic of user by mosaic_id
    try:
        res = mosaic_retrieve_service_by_id(username=credentials.username, mosaic_id=mosaic_id)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic/{mosaic_id}/image GET
@router.get(ROUTE_PREFIX + "/mosaic/{mosaic_id}/image", status_code=status.HTTP_200_OK,
            response_model=Union[List[DataReturnSchema], DataReturnErrorSchema])
def get_mosaic_images(request: Request, response: Response,
                      mosaic_id: UUID,
                      credentials: HTTPCredentialsCustom = Depends(security)):
    # List all images in mosaic
    try:
        res = get_mosaic_images_service(username=credentials.username, mosaic_id=mosaic_id)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic/{mosaic_id}/image/{image_id} GET
@router.get(ROUTE_PREFIX + "/image/{image_id}", status_code=status.HTTP_200_OK,
            response_model=Union[DataReturnSchema, DataReturnErrorSchema])
def get_image_by_id(request: Request, response: Response, image_id: UUID,
                    credentials: HTTPCredentialsCustom = Depends(security)):
    # get single image by image_id from mosaic_id
    try:
        res = get_image_by_id_service(username=credentials.username, image_id=image_id)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic/{mosaic_id} DELETE
@router.delete(ROUTE_PREFIX + "/mosaic/{mosaic_id}", status_code=status.HTTP_200_OK)
def mosaic_delete(request: Request, response: Response, mosaic_id: UUID,
                  credentials: HTTPCredentialsCustom = Depends(security)):
    """
    Delete existing mosaic and all files
    :return: {"mosaic_id": "delete OK"}
    """
    try:
        res = mosaic_delete_service(mosaic_id, credentials.username)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic?tags=some_tag&tags=another_tag GET
@router.get(ROUTE_PREFIX + "/mosaic", status_code=status.HTTP_200_OK, response_model=List[MosaicReturnSchema])
def get_mosaic(request: Request, response: Response,
               tags: Optional[List[str]] = Query(default=None),
               credentials: HTTPCredentialsCustom = Depends(security)):
    """
    get all mosaics of user.
    tags are optional
    In case, if tags are defined, return only mosaics that have specified tags
    """
    try:
        res = mosaic_retrieve_service(username=credentials.username, tags=tags)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


@router.delete(ROUTE_PREFIX + "/image/{image_id}", status_code=status.HTTP_200_OK)
def delete_image_by_id(request: Request, response: Response, image_id: UUID,
                       credentials: HTTPCredentialsCustom = Depends(security)):
    # delete image_id
    try:
        res = delete_image_by_id_service(username=credentials.username, image_id=image_id)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500)
    return res


# /mosaic/image POST
@router.post(ROUTE_PREFIX + "/mosaic/image", status_code=status.HTTP_200_OK,
             dependencies=[Depends(user_memory_limit_update)])
def create_mosaic_and_file_upload(request: Request, response: Response,
                                  name: str,
                                  tags: List[str] = Query(),  # default=None),
                                  credentials: HTTPCredentialsCustom = Depends(security),
                                  file: UploadFile = File(...)):
    try:
        # create new mosaic and upload file/files to it
        res = create_mosaic_and_file_upload_service(username=credentials.username, files=[file],
                                                    tags=tags, name=name)
        return res
    except (MemoryLimitExceeded, FileTooBig) as e:
        logger.debug(f'User: {credentials.username}, error: {str(e)}')
        raise HTTPException(e.http_code, e.detail())
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(500)


# /mosaic/{mosaic_id}/image POST
@router.post(ROUTE_PREFIX + "/mosaic/{mosaic_id}/image", status_code=status.HTTP_200_OK,
             dependencies=[Depends(user_memory_limit_update)])
def upload_image_to_existing_mosaic(request: Request, response: Response,
                                    mosaic_id: UUID,
                                    credentials: HTTPCredentialsCustom = Depends(security),
                                    file: UploadFile = File(...)):
    try:
        res = upload_images_to_existing_mosaic_service(username=credentials.username,
                                                       mosaic_id=mosaic_id,
                                                       files=[file])
        return res
    except (MemoryLimitExceeded, FileTooBig) as e:
        # Normal behavior - user tried to upload too much
        logger.debug(f'User: {credentials.username}, error: {str(e)}')
        raise HTTPException(e.http_code, e.detail())
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(500)


# /mosaic/{mosaic_id}/link-image
@router.post(ROUTE_PREFIX + '/mosaic/{mosaic_id}/link-image', status_code=status.HTTP_200_OK,
             dependencies=[Depends(user_memory_limit_update)])
def link_image_from_minio_to_mosaic(request: Request, response: Response,
                                    json: LinkImageSchema, mosaic_id: UUID,
                                    credentials: HTTPCredentialsCustom = Depends(security)):
    try:
        res = link_image_from_minio_to_mosaic_service(username=credentials.username,
                                                      mosaic_id=mosaic_id,
                                                      image_url=json.url)
        return res
    except (MemoryLimitExceeded, FileTooBig) as e:
        # Normal behavior - user tried to upload too much
        logger.debug(f'User: {credentials.username}, error: {str(e)}')
        raise HTTPException(e.http_code, e.detail())
    except DataCatalogError as e:
        # Unexpected behavior; logging exception details
        logger.exception(f'Exception for mosaic/link_image request for user {credentials.username}: str(e)')
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(500)


# /legacy_whitemaps/upload
@router.post("/rest/rasters", status_code=status.HTTP_200_OK, dependencies=[Depends(user_memory_limit_update)])
def legacy_whitemaps_upload(request: Request, response: Response,
                            credentials: HTTPCredentialsCustom = Depends(security),
                            file: UploadFile = File(...)):
    # legacy API to mimic WM api
    try:
        res = legacy_whitemaps_upload_service(username=credentials.username, file=file, tags=[])
        return res
    except (MemoryLimitExceeded, FileTooBig) as e:
        # Normal behavior - user tried to upload too much
        logger.debug(f'User: {credentials.username}, error: {str(e)}')
        raise HTTPException(e.http_code, e.detail())
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(str(e))
        raise HTTPException(500)

