from fastapi import status
from errors.base_error import DataCatalogError


class ItemNotFound(DataCatalogError):
    def __init__(self, uid, instance_type=None, http_code=status.HTTP_404_NOT_FOUND):
        instance_type = instance_type or 'Item'
        super().__init__(message=f"{instance_type} {uid} not found",
                         uid=str(uid),
                         http_code=http_code)


class AccessDenied(DataCatalogError):
    def __init__(self, uid, user, instance_type=None, http_code=status.HTTP_403_FORBIDDEN):
        instance_type = instance_type or "item"
        super().__init__(message=f"Access to {instance_type} {uid} denied for {user}! ",
                         uid=str(uid),
                         http_code=http_code)


class InternalError(DataCatalogError):
    def __init__(self,
                 username=None,
                 mosaic_id=None,
                 image_id=None,
                 service=None,
                 http_code=status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message="Internal error. user: {username}, "
                                 "mosaic_id: {mosaic_id}, "
                                 "image_id: {image_id}, "
                                 "service: {service}",
                         username=username,
                         mosaic_id=str(mosaic_id),
                         image_id=str(image_id),
                         service=service,
                         http_code=http_code)


class FileAlreadyExists(DataCatalogError):
    def __init__(self, url, http_code=status.HTTP_403_FORBIDDEN):
        super().__init__(message="File already exists inside mosaic. Url: {url}",
                         url=url,
                         http_code=http_code)


class PreviewNotFound(DataCatalogError):
    def __init__(self, image_id, http_code=status.HTTP_404_NOT_FOUND):
        super().__init__(message="Preview not found for image: {image_id}",
                         image_id=str(image_id),
                         http_code=http_code)
