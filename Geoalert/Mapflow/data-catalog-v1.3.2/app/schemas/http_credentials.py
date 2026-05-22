from pydantic import BaseModel


class HTTPCredentialsCustom(BaseModel):
    user_id: str
    username: str
    is_admin: bool = False  # added is_admin field, so handlers requiring admin permissions can use it
