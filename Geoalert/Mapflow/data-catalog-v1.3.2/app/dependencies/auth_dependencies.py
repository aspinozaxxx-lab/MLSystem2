import binascii
from base64 import b64decode
from typing import Optional
from loguru import logger
import asyncio
import aiohttp
from jose import jwt
from fastapi import Request, HTTPException
from fastapi.openapi.models import HTTPBase
from fastapi.security.base import SecurityBase
from fastapi.security.utils import get_authorization_scheme_param
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from starlette.authentication import AuthenticationError

from schemas.http_credentials import HTTPCredentialsCustom
from crud.user import add_or_update_user
from config import Config

# Authentication server url (test server by default)
AUTH_URL = Config.AUTH_HOST

class StandardHTTPSecurity(SecurityBase):
    """
    Dependency for authenticating user by basic auth or bearer token.
    Also, this dependency extracts username/email from token, and passes it for further processing.
    Now supports extracting from HTTPBasic auth scheme and bearer scheme.
    """
    def __init__(self, *,
                 scheme_name: Optional[str] = None,
                 realm: Optional[str] = None,
                 description: Optional[str] = None,
                 auto_error: bool = True):
        description = "Standard HTTP authentication systems, including: basic auth, bearer or httpdigest"
        self.model = HTTPBase(scheme="basic", description=description)  # this model is used by OpenAPI
        self.scheme_name = scheme_name or self.__class__.__name__
        self.realm = realm
        self.auto_error = auto_error
        self.jwt_key = None
        self.x_actor_id = Config.X_ACTOR_ID

    async def _fetch_jwt_key(self):
        """
         JWT_KEY is returned from open API
         We fetch it on initialization, and then refresh in case token validation fails
        """
        url = Config.PUBLIC_KEY_URL
        logger.debug("Requesting jwt public key")
        async with aiohttp.ClientSession() as session:
            async with session.get(url,
                                   timeout=10,
                                   proxy=None,
                                   raise_for_status=True) as response:
                res = await response.json()
        logger.debug(f"Got jwt key response: {res}")
        return self._parse_jwt_key_response(res)

    @staticmethod
    def _parse_jwt_key_response(res):
        return res['keys'][0]

    async def _fetch_group_ids(self, user_id, x_actor_id):
        url = Config.USER_INFO_URL
        async with aiohttp.ClientSession() as session:
            async with session.get(url.format(user_id),
                                   timeout=10,
                                   proxy=None,
                                   headers={"x-actor-id": f"{x_actor_id}"},
                                   raise_for_status=True) as response:
                res = await response.json()
        return self._parse_group_ids_response(res)

    @staticmethod
    def _parse_group_ids_response(res):
        return [group['id'] for group in res.get('userGroups', [])]

    async def __call__(self, request: Request) -> HTTPCredentialsCustom:
        authorization: str = request.headers.get("Authorization")

        # scheme: 'Bearer'; param: token'
        scheme, param = get_authorization_scheme_param(authorization)
        logger.debug(f"Auth scheme: {scheme}")
        unauthorized_headers = {f"WWW-Authenticate": f"{scheme}"}
        if not authorization or scheme.lower() != 'bearer':
            raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers=unauthorized_headers,
        )
        return await self._authenticate_jwt(token=

    async def _authenticate_jwt(self, token:  -> HTTPCredentialsCustom:
        try:
            decoded =  jwt.decode(token,
                                  self.jwt_key,
                                  audience=Config.OIDC_CLIENT,
                                  options={'verify_aud': False})
        except Exception as e:
            # re-fetch key and alg, and try again
            logger.info(f"Decode token error: {e}, requesting jwt public key")
            self.jwt_key = await self._fetch_jwt_key()
            logger.debug(f"Fetched key for retry: JWT key {self.jwt_key}")
            try:
                decoded = jwt.decode(token,
                                     self.jwt_key,
                                     audience=Config.OIDC_CLIENT,
                                     options={'verify_aud': False})
            except Exception as exc:
                logger.exception(f"Error in jwt token decode: {str(exc)}")
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authenticated")

        try:
            user_id = decoded.get('uid')
            logger.debug(decoded)
            # Email may be absent in the token, and there is no guarantee that any other thing is unique
            # In original data-catalog we use email as user identification,
            # so now we just write id into the email (duplicate) to not change this behavior
            email = decoded.get('uid')
            group_ids = await self._fetch_group_ids(user_id=user_id, x_actor_id=self.x_actor_id)
            logger.debug(f"{group_ids=}")
            is_admin = any(group_id in Config.ADMIN_GROUP_IDS for group_id in group_ids)
            is_user = any(group_id in Config.USER_GROUP_IDS for group_id in group_ids)

            if is_admin:
                logger.debug(f'User {email} authenticated as admin')
                add_or_update_user(is_admin=True, user_id=user_id, email=email)
                return HTTPCredentialsCustom(user_id=user_id, username=email, is_admin=True)
            elif is_user:
                logger.debug(f'User {email} authenticated as user, not admin')
                add_or_update_user(is_admin=False, user_id=user_id, email=email)
                return HTTPCredentialsCustom(user_id=user_id, username=email, is_admin=False)
            else:
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Forbidden")
        except HTTPException as e:
            raise e
        except Exception as e:
            # todo: Actually, here we should dive into jwt exceptions and use the particular class,
            # but the documentation is unclear for this case, so by now leaving this as is
            logger.exception(f"Error in jwt token decode for user {email}: {str(e)}")
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Not authenticated")
