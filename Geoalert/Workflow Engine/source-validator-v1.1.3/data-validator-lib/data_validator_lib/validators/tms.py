from loguru import logger
from typing import Tuple, Optional
from string import Formatter
from ..base.status import Status
from .basemap import BasemapValidator, URL_KEY
from ..base.error_message import ErrorMessage
from ..errors import tms as tms_error, basemap as basemap_error


class TMSValidator(BasemapValidator):
    def _request_is_ok(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Checks if the request contains valid url, that is string starting with http:// or https://
        and contains {x}, {y}, {z} placeholders
        Args:
            request: data request

        Returns:
            Status.Error and description if request is not OK, Status.OK, {} otherwise
        """
        # check if request['zoom'] is valid
        status, description = self._zoom_is_valid(request)
        if status == Status.ERROR:
            return status, description
        source_type = request.get('source_type', None)
        if source_type != 'tms':
            raise RuntimeError(f'TMS validator called with wrong {source_type = }')
        status, description = super()._request_url_is_valid(request)
        if status == Status.ERROR:
            # we do not need to check further if it is already not a valid basemap request
            return status, description

        url = request.get(URL_KEY, None)
        try:
            format_keys = [entry[1] for entry in Formatter().parse(url) if entry[1] is not None]
            # see https://stackoverflow.com/questions/46161710/how-to-check-if-string-has-format-arguments-in-python
        except ValueError as e:
            logger.exception("Error while parsing image url for format keys:")
            return Status.ERROR, basemap_error.UrlFormatError(str(e))

        if not ['x', 'y', 'z'] == sorted(format_keys, reverse=False):
            return Status.ERROR, tms_error.TMSLinkFormatError()
        else:
            # if super().... will return Status.WARN, we need to return it in case no ERRORs were found here
            return status, description
