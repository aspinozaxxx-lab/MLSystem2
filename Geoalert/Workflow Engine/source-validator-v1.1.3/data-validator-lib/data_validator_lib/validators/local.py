from loguru import logger
from typing import Tuple, Iterable, Optional, Union, List
from botocore.exceptions import BotoCoreError, ClientError

from ..base.status import Status
from ..base.validator import Validator, BadRequirements
from ..functional import read_profile_from_s3, get_pixel_size, DEFAULT_IMAGE_SIZE, is_tiff, is_folder

from ..base.error_message import ErrorMessage
from ..errors import local as local_error
from ..base.constants import METADATA_KEY, S3_LINK_KEY, REQUIRED_METADATA_KEYS, RES_TOLERANCE


class LocalValidator(Validator):
    def __init__(self, storage=None, **kwargs):
        super().__init__(**kwargs)
        self.storage = storage

    # ================ check request section ==================== #
    @staticmethod
    def _metadata_request_is_ok(metadata: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Check that image metadata has necessary keys - crs, transform, number of channels and data type
        """
        if type(metadata) != dict:
            return Status.ERROR, local_error.ImageMetadataMustBeDict()
        if not REQUIRED_METADATA_KEYS <= set(metadata.keys()):
            return Status.ERROR, local_error.ImageMetadataKeyError()
        return Status.OK, None

    def _file_link_is_ok(self, s3_link: str) -> Tuple[Status, Optional[ErrorMessage]]:
        """
            Checks if file link corresponds minio link format
        """
        if type(s3_link) != str:
            return Status.ERROR, local_error.S3URLError(s3_link)
        if not s3_link.startswith('s3://'):
            # maybe add more link validation?
            return Status.ERROR, local_error.S3URLError(s3_link)
        if not is_tiff(s3_link) and not is_folder(s3_link):
            return Status.ERROR, local_error.S3URLError(s3_link)
        return Status.OK, None

    def _request_is_ok(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Check if request corrsponds the format.
        The request for local file must contain either image metadata or file link to download file and check it
        """
        # using image is preferrable so that user could not pass wrong metadata along with the image
        s3_link = request.get(S3_LINK_KEY, None)
        if s3_link:
            return self._file_link_is_ok(s3_link)
        # if s3_link is not provided, ve search for the metadata dictionary inside the request
        metadata = request.get(METADATA_KEY, None)
        if metadata:
            return self._metadata_request_is_ok(metadata)

        return Status.ERROR, local_error.LocalRequestKeyError()

    # ================== Check params section ====================== #
    def _check_file_params(self, requirements: dict, s3_link: str) -> bool:
        """
        get metadata from file at s3 and then check the metadata against the requirements
        """
        if not self.storage:
            raise RuntimeError('Validator is not connected to any s3 resource and cannot validate local files')
        try:
            metadata = read_profile_from_s3(self.storage, s3_link)
            if not REQUIRED_METADATA_KEYS.issubset(set(metadata.keys())):
                self.params_message.append(local_error.BadImageProfile(profile=metadata,
                                                                       required_keys=list(REQUIRED_METADATA_KEYS)))
                return False
        except (BotoCoreError, ClientError) as e:
            logger.exception("Error while copying file from s3:")
            self.params_message.append(local_error.ReadFromS3Failed(s3_link=s3_link, message=str(e)))
        except Exception as e:
            logger.exception("Error while opening file and reading metadata:")
            self.params_message.append(local_error.ImageReadError(message=str(e)))
            return False
        else:
            logger.debug(f"Read profile from s3: {metadata}")
            return self._check_metadata_params(requirements, metadata)

    def _check_dtype(self, required_dtypes: Optional[Union[list, str]], request_dtype: str) -> bool:
        """
        checks if request_dtype (taken from processing request) is allowed in required_dtypes
        'None' required_dtypes means that any is allowed
        Args:
            required_dtypes: either single dtype (string) or a list of allowed dtypes
            request_dtype: the dtype to check (string)
        """
        if type(required_dtypes) == str:
            required_dtypes = [required_dtypes]
        if isinstance(required_dtypes, List):
            if not request_dtype in required_dtypes:
                self.params_message.append(local_error.DtypeNotAllowed(required_dtypes=required_dtypes,
                                                                       request_dtype=request_dtype))
                return False
        elif required_dtypes is not None:
            raise BadRequirements('Dtype Requirements, if present, must be either String for a single dtype '
                                  f'or a List of allowed dtypes. Got type {type(required_dtypes)} ')
            # several allowed dtypes
        return True

    def _check_resolution(self, min_res: Optional[float], max_res: Optional[float],
                          crs, transform,
                          width: int = DEFAULT_IMAGE_SIZE, height: int = DEFAULT_IMAGE_SIZE,
                          res_tolerance: float = RES_TOLERANCE) -> bool:
        """
        Extracts real spatial resolution (gsd) from image params and checks if it meets the requirements.
        There is tolerance (defaults to 0.1), which means that the resolution can be 10% less than min_res
        or 10% more than max_res. Otherwise, discrrepancy from reprojection can lead to reject of good data\

        in case of error, sets the 'res' section of params_message

        crs and transform are not type-hinted because it is up to rasterio/affine which input would they understand
        see functional/pixel_size.py where these parameters are passed to rasterio.warp functions
        """
        res = get_pixel_size(transform, crs, width, height)
        if min_res and res*(1 + res_tolerance) < min_res:
            self.params_message.append(local_error.PixelSizeTooLow(actual_res=res,
                                                                           min_res=min_res))
            return False
        elif max_res and max_res < res/(1 + res_tolerance):
            self.params_message.append(local_error.PixelSizeTooHigh(actual_res=res,
                                                                            max_res=max_res))
            return False
        return True

    def _check_nchannels(self, required_nchannels: Union[List[int], int], count: int) -> bool:
        if isinstance(required_nchannels, int):
            required_nchannels = [required_nchannels]
        if isinstance(required_nchannels, List):
            if count not in required_nchannels:
                self.params_message.append(local_error.NChannelsNotAllowed(required_nchannels=required_nchannels,
                                                                           real_nchannels=count))
                return False
        elif required_nchannels is not None:
            raise BadRequirements('nchannels Requirements, if present, must be either Int for a single channels number '
                                  f'or a List of allowed nchannels. Got type {type(required_nchannels)} ')
        return True

    def _check_metadata_params(self, requirements: dict, metadata: dict) -> bool:
        """
        check the metadata against the requirements.
        Metadata must be in format that is retrieved from rasterio.open().profile
        Despite it would be enough one False check, we will go through all of them to fill all the params_messages
        """
        status = True

        # dtype
        required_dtypes = requirements.get('dtype', None)
        try:
            status = self._check_dtype(required_dtypes, metadata['dtype']) and status
        except Exception as e:
            logger.exception('Error while checking dtype:')
            self.params_message.append(local_error.ImageCheckError(checked_param="dtype",
                                                                   metadata=metadata,
                                                                   err_message=str(e)))
            status = False
        # spatial resolution
        min_res = requirements.get('min_res', None)
        max_res = requirements.get('max_res', None)
        try:
            status = self._check_resolution(min_res,
                                            max_res,
                                            metadata['crs'],
                                            metadata['transform'],
                                            metadata['width'],
                                            metadata['height']) and status
        except Exception as e:
            logger.exception('Error while checking spatial resolution:')
            self.params_message.append(local_error.ImageCheckError(checked_param="spatial resolution",
                                                                   metadata=metadata,
                                                                   err_message=str(e)))
            status = False
        # nchannels
        nchannels = requirements.get('nchannels', None)
        try:
            status = self._check_nchannels(nchannels, metadata['count']) and status
        except Exception as e:
            logger.exception('Error while checking number of channels:')
            self.params_message.append(local_error.ImageCheckError(checked_param="number of channels",
                                                                   metadata=metadata,
                                                                   err_message=str(e)))
            status = False
        return status

    def _check_params(self, requirements: dict, request: dict) -> bool:
        """
        Checks that the file params meet the requirements. The parameters are: dtype, crs, resolution
        """
        s3_link = request.get(S3_LINK_KEY, None)
        if s3_link:
            if is_tiff(s3_link):
                return self._check_file_params(requirements, s3_link)
            elif is_folder(s3_link):
                files = self.storage.list_files(s3_link, extensions=('tif', 'tiff'))
                if not files:
                    self.params_message.append(local_error.EmptyFolder(s3_link=s3_link))
                    return False
                for file in files:
                    result = self._check_file_params(requirements, file)
                    if not result:
                        # If any of files is not acceptable, fail immediately
                        # All other messages (warnings) are appended for all the files
                        # todo: aggregation of similar/same messages from multiple files?
                        return False
                return True
            else:
                raise RuntimeError('s3 link must have been checked earlier')
        metadata = request.get(METADATA_KEY, None)
        if metadata:
            return self._check_metadata_params(requirements, metadata)

        raise RuntimeError('Existence of s3_link or metadata must have been checked earlier!')