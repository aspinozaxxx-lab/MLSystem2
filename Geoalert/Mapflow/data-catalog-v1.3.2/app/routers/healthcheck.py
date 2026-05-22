from fastapi import APIRouter, HTTPException, status

from crud.misc import db_healthcheck
from config import Config

router = APIRouter()

ROUTE_PREFIX = Config.ROUTE_PREFIX


# /heartbeat/lite
# /tmp - healthcheck
@router.get("/heartbeat/lite", status_code=status.HTTP_200_OK)
def healthcheck():
    return 'OK'


@router.get("/heartbeat", status_code=status.HTTP_200_OK)
def healthcheck_db():
    if db_healthcheck():
        return 'OK'
    else:
        raise HTTPException(500)
