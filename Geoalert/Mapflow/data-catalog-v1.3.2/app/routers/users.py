from config import Config
from fastapi import APIRouter, Request, Depends

from dependencies.auth_dependencies import StandardHTTPSecurity
from schemas.http_credentials import HTTPCredentialsCustom

security = StandardHTTPSecurity()

router = APIRouter()

ROUTE_PREFIX = Config.ROUTE_PREFIX


@router.get(ROUTE_PREFIX + "/users/")
def list_users(request: Request, credentials: HTTPCredentialsCustom = Depends(security)):
    """
    Route to list all users in DB
    Admins can list all users in DB and get all information about their data
    """

    return ["users"] if request.user.is_authenticated else ['not authenticated']


@router.get(ROUTE_PREFIX + "/users/{user_id}")
def get_user(request: Request, user_id: str, credentials: HTTPCredentialsCustom = Depends(security)):
    """
    Route to get the specified user data
    Get all the information about user data
    HTTP header contains the information about which user is doing request (login and password)
    {user_id} - is the id of the user, data of which is requested
    middleware checks, if user requesting the data is admin. if admin, return all data of the {user_id}
    if not admin, but authorization is ok, returns user data
        else data-catalog returns not-authorized error
    """
    return user_id


# TODO do we really need memory limit update?
# maybe admins can update memory limit of the user ?
@router.post(ROUTE_PREFIX + "/users/{user_id}")
def update_user(request: Request, user_id: str, memory_limit: int):
    """
    Route to update user's data preserving all his artifacts
    Args:
        user_id: user ID to update
        memory_limit: memory limit of a user
    Returns:

    """
    return user_id
