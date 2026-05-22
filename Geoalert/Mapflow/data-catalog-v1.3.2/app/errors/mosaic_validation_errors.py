from fastapi import status
from errors.base_error import DataCatalogError


class FileValidationFailed(DataCatalogError):
    def __init__(self,
                 mosaic_id,
                 filename,
                 param_name,
                 got_param,
                 expected_param,
                 http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        super().__init__(message="File: {filename} can't be uploaded to mosaic: {mosaic_id}. "
                                 "param_name: {param_name}, got_param: {got_param}, expected_param: {expected_param}",
                         filename=filename, mosaic_id=str(mosaic_id), http_code=http_code,
                         param_name=param_name, got_param=got_param, expected_param=expected_param)

