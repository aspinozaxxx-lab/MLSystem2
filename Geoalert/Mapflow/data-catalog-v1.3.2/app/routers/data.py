from uuid import UUID

from fastapi import APIRouter, Request, status, Depends, HTTPException

from loguru import logger

from errors import DataCatalogError
from schemas.data import MemoryInfoOfUserSchema
from schemas.http_credentials import HTTPCredentialsCustom
from service.data import get_memory_info_of_user_service, get_image_preview_service
from dependencies.auth_dependencies import StandardHTTPSecurity
from config import Config

router = APIRouter()

security = StandardHTTPSecurity()

ROUTE_PREFIX = Config.ROUTE_PREFIX


@router.get(ROUTE_PREFIX + "/memory", status_code=status.HTTP_200_OK, response_model=MemoryInfoOfUserSchema)
def get_memory_info_of_user(request: Request, credentials: HTTPCredentialsCustom = Depends(security)):
    try:
        res = get_memory_info_of_user_service(login=credentials.username)
        return res
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get(ROUTE_PREFIX + "/image/{image_id}/preview/{preview_type}", status_code=status.HTTP_200_OK)
def get_image_preview(request: Request, image_id: UUID, preview_type: str,
                      credentials: HTTPCredentialsCustom = Depends(security)):
    try:
        return get_image_preview_service(login=credentials.username, image_id=image_id, preview_type=preview_type)
    except DataCatalogError as e:
        logger.exception(str(e))
        raise HTTPException(e.http_code, e.detail())
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)
