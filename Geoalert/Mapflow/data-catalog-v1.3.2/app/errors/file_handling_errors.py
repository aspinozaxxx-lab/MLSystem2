from fastapi import status

from config import Config
from errors.base_error import DataCatalogError


class MemoryLimitExceeded(DataCatalogError):
    def __init__(self,
                 available_memory,
                 memory_requested,
                 http_code=status.HTTP_403_FORBIDDEN):
        super().__init__(message="Available disk space remained: {available_memory}, "
                                 "requested memory: {memory_requested}",
                         http_code=http_code,
                         available_memory=available_memory,
                         memory_requested=memory_requested)
        self.available_memory = available_memory


class FileTooBig(DataCatalogError):
    def __init__(self,
                 actual_file_size,
                 max_file_size=Config.MAX_UPLOAD_FILE_SIZE,
                 http_code=status.HTTP_403_FORBIDDEN):
        super().__init__(message="Max upload file size limit exceeded."
                                 "Max upload file size = {max_file_size} bytes, \
                                 got file size = {actual_file_size} bytes",
                         http_code=http_code,
                         max_file_size=max_file_size,
                         actual_file_size=actual_file_size)
        self.actual_file_size = actual_file_size


class FileCheckFailed(DataCatalogError):
    def __init__(self,
                 filename, bad_parameters,
                 crs, transform, width, height,
                 http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        # This is a bit awkward message, but we need to put all the parameters in one error
        # to report all the problems at onc
        message = "File {filename} is not suitable for processing. It must have valid CRS and transform" \
                  "and the dimensions must be less than " + str(Config.MAX_IMAGE_SIZE_PIXELS) + \
                  ". It has parameters: crs = {crs}, transform = {transform}," \
                  "width={width}, height={height}. " \
                  "The following parameters do not meet the requirements: {bad_parameters}"
        super().__init__(message=message,
                         http_code=http_code,
                         filename=filename,
                         bad_parameters=bad_parameters,
                         crs=str(crs), transform=str(transform), width=width, height=height
                         )


class FileOpenError(DataCatalogError):
    def __init__(self,
                 filename,
                 http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        super().__init__(message="File {filename} cannot be open as Geotiff file",
                         http_code=http_code,
                         filename=filename)


class ImageOutOfBounds(DataCatalogError):
    def __init__(self,
                 http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        super().__init__(message="File can't be uploaded, because its extent is out of coordinate range."
                                 "Most probably, CRS and/or transform are invalid",
                         http_code=http_code)


class ImageExtentTooBig(DataCatalogError):
    def __init__(self,
                 http_code=status.HTTP_422_UNPROCESSABLE_ENTITY):
        super().__init__(message="File can't be uploaded, because the geometry of the image is too big,"
                                 " we will not be able to process it properly",
                         http_code=http_code)