from typing import Tuple, List
from loguru import logger
from data_validator_lib import get_validator
from data_validator_lib.base.factory import source_types
from data_validator_lib.base.status import Status
from data_validator_lib.base.error_message import ErrorMessage, InternalError
from data_validator_lib.errors.validator import RequestMustHaveSourceType, SourceTypeIsNotAllowed


def validate(wd: dict, request: dict, **validator_kwargs) -> Tuple[Status, List[ErrorMessage]]:
    source_type = request.get('source_type', None)
    if source_type is None:
        return Status.ERROR, [RequestMustHaveSourceType()]
    try:
        validator = get_validator(source_type, **validator_kwargs)
    except ModuleNotFoundError as e:
        logger.exception(f"Wrong validator type requested: {source_type}.")
        return Status.ERROR, [SourceTypeIsNotAllowed(source_type=source_type, allowed_sources=source_types)]
    try:
        status, description = validator(wd, request)
    except Exception as e:
        logger.exception(f"Unhandled error in {type(validator)} during validation.")
        return Status.ERROR, [InternalError(exception=e)]
    return status, description
