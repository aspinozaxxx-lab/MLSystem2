from fastapi import status
from errors.base_error import DataCatalogError


class InvalidLinkToMinio(DataCatalogError):
    def __init__(self, object_url, http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        super().__init__(message="Invalid image url for minio provided. Url: {object_url}",
                         object_url=object_url)


class MinioObjectDoesntExist(DataCatalogError):
    def __init__(self, object_url, http_code=status.HTTP_404_NOT_FOUND):
        super().__init__(message="Object doesn't exists. Url: {object_url}",
                         object_url=object_url)
