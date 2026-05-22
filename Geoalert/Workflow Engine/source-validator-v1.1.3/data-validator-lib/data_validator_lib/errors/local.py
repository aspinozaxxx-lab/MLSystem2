from ..base.error_message import ErrorMessage
from ..base.constants import REQUIRED_METADATA_KEYS, S3_LINK_KEY, METADATA_KEY


class ImageMetadataMustBeDict(ErrorMessage):
    def __init__(self):
        super().__init__(message="Image metadata must be a dict (json)")


class ImageMetadataKeyError(ErrorMessage):
    def __init__(self):
        super().__init__(message=f"Image metadata must have keys: {list(REQUIRED_METADATA_KEYS)}")


class S3URLError(ErrorMessage):
    def __init__(self, actual_s3_link):
        super().__init__(message="URL of the image at s3 storage must be a string starting with s3://, "
                                     "and must be either folder ending with `/` or .tif/.tiff file "
                                     "got {actual_s3_link}",
                         actual_s3_link=actual_s3_link)

class ImageMustBeTiff(ErrorMessage):
    def __init__(self, actual_s3_link):
        super().__init__(message="Image must be a tiff file, got url {actual_s3_link}",
                         actual_s3_link=actual_s3_link)

class LocalRequestKeyError(ErrorMessage):
    def __init__(self):
        super().__init__(message=f"Request must contain either `{METADATA_KEY}` or `{S3_LINK_KEY}` keys")


class ReadFromS3Failed(ErrorMessage):
    def __init__(self, s3_link, message):
        # error message is passed only to logs, not to user
        super().__init__(message=f"Failed to read file from {{s3_link}}, error: {message}",
                         s3_link=s3_link)


class ImageReadError(ErrorMessage):
    def __init__(self, message):
        # error message is passed only to logs, not to user
        super().__init__(message=f"Failed to open the file or read metadata, error: {message}")


class DtypeNotAllowed(ErrorMessage):
    def __init__(self, required_dtypes, request_dtype):
        super().__init__(message="Dtype must be one of {required_dtypes}, got {request_dtype}",
                         required_dtypes=required_dtypes,
                         request_dtype=request_dtype)


class PixelSizeTooLow(ErrorMessage):
    def __init__(self, actual_res, min_res):
        super().__init__(code="SpatialResolutionTooLow",
                         message="Spatial resolution is too high. "
                                     "Got {actual_res}, minimum allowed pixel size is {min_res}",
                         actual_res=actual_res,
                         min_res=min_res)


class PixelSizeTooHigh(ErrorMessage):
    def __init__(self, actual_res, max_res):
        super().__init__(message="Spatial resolution is too low. "
                                     "Got {actual_res}, maximum allowed pixel size is {max_res}",
                         actual_res=actual_res,
                         max_res=max_res)


class NChannelsNotAllowed(ErrorMessage):
    def __init__(self, required_nchannels, real_nchannels):
        super().__init__(message=f"Number of channels must be {required_nchannels}, got {real_nchannels}",
                         required_nchannels=required_nchannels,
                         real_nchannels=real_nchannels)


class BadImageProfile(ErrorMessage):
    def __init__(self, profile, required_keys):
        super().__init__(message="Image profile (metadata) must have keys {required_keys},"
                                     "got profile: {profile}",
                         profile=profile,
                         required_keys=required_keys)


class ImageCheckError(ErrorMessage):
    def __init__(self, checked_param, metadata, err_message):
        super().__init__(message="Error occurred during image {checked_param} check: {err_message}. "
                                     "Image metadata = {metadata}",
                         checked_param=checked_param,
                         metadata=metadata,
                         err_message=err_message)

class EmptyFolder(ErrorMessage):
    def __init__(self, s3_link):
        super().__init__(message="S3 folder object `{s3_link}` does not contain any images",
                         s3_link=s3_link)