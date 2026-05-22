from loguru import logger
import requests
from fastapi import Request, HTTPException, status

from crud.user import user_update
from config import Config

AUTH_URL = Config.AUTH_HOST

def user_memory_limit_update(request: Request):
    # Return immediately if we do not check memory limit - to avoid dependencty on Mapflow API if it is not needed
    if not Config.MEMORY_LIMIT:
        return

    auth_header = request.headers.get('Authorization')
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    try:
        response = requests.get(url=AUTH_URL, headers={'Authorization': auth_header})
        if response.status_code != 200:
            logger.warning(f"Request to whitemaps for user status doesn't succeed. Response: {response.json()}, "
                           f"status_code: {response.status_code}")
            raise HTTPException(response.status_code)
        is_admin = response.json().get('isAdmin')
        login = response.json().get('email')
        memory_limit = response.json().get('memoryLimit')
        if not memory_limit or not login:
            raise HTTPException(500)
        user_update(isAdmin=is_admin, login=login, memoryLimit=memory_limit)
    except HTTPException as e:
        raise e
    except Exception as e:
        # if something goes wrong
        logger.exception(str(e))
        raise HTTPException(500)
    return

