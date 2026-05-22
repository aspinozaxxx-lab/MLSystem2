import re
from typing import Tuple, Optional, List
from ..base.validator import Validator, BadRequirements
from ..base.status import Status
from ..base.error_message import ErrorMessage
from ..errors import basemap as basemap_error
from ..base.constants import ZOOM_KEY, URL_KEY, LOWEST_ZOOM, HIGHEST_ZOOM, BASEMAPS_BLACKLIST_PATTERNS


class BasemapValidator(Validator):
    """
    Base class for xyz, tms, wms validation, because all of them are tile
    services with zoom-value, so part of validation may be the same

    checks the request params: zoom to be in the specified range
    """
    @staticmethod
    def _url_in_blacklist(blacklist: List[re.Pattern], url: str):
        for pattern in blacklist:
            if pattern.search(url):
                return pattern.pattern
        return False

    def _request_url_is_valid(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Checks if the request contains valid url, that is string starting with http:// or https://
        Args:
            request: data request

        Returns:
            Status.Error and description if request is not OK, Status.OK, {} otherwise
        """
        url = request.get(URL_KEY, None)
        if type(url) != str:
            return Status.ERROR, basemap_error.UrlMustBeString(type(url))
        if not (url.startswith('http://') or url.startswith('https://')):
            return Status.ERROR, basemap_error.UrlMustBeLink()
        pattern = self._url_in_blacklist(BASEMAPS_BLACKLIST_PATTERNS, url)
        if pattern:
            return Status.ERROR, basemap_error.UrlBlacklisted(url, pattern)
        return Status.OK, None

    def _request_is_ok(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        raise NotImplementedError

    def _zoom_is_ok(self, requirements: dict, request: dict) -> bool:
        """
        check if zoom in the parameters matches the requirements.
        Actually, this parameter is currently not allowed in API, but it will be.
        Args:
            requirements: part of the requirements from WD
            request: request dict

        Returns:
            True if zoom meets the requirements, False otherwise
        """
        zoom = request.get(ZOOM_KEY, None)
        if zoom is None:
            # if zoom is not specified, it will be taken from WD
            return True
        # zoom must be casted to int
        try:
            zoom = int(zoom)
        except (ValueError, TypeError) as e:
            self.params_message.append(basemap_error.ZoomMustBeInteger(actual_zoom=zoom))
            return False
        # if there is no explicit requirements in the wd, we use the default range [LOWEST_ZOOM, HIGHEST_ZOOM]
        min_zoom = requirements.get('min_zoom', LOWEST_ZOOM)
        max_zoom = requirements.get('max_zoom', HIGHEST_ZOOM)
        if min_zoom > max_zoom:
            raise BadRequirements(f'{min_zoom = } must be lower than the {max_zoom = }')
        if zoom < min_zoom:
            self.params_message.append(basemap_error.TooLowZoom(actual_zoom=zoom,
                                                                min_zoom=min_zoom))
            return False
        elif zoom > max_zoom:
            self.params_message.append(basemap_error.TooHighZoom(actual_zoom=zoom,
                                                                 max_zoom=max_zoom))
            return False
        return True

    def _zoom_is_valid(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        # check if request['zoom'] is OK
        zoom = request.get('zoom', None)
        # if zoom is None, skip checks, and it will be taken from WD
        if zoom is not None:
            try:
                zoom = int(zoom)
            except (ValueError, TypeError) as e:
                return Status.ERROR, basemap_error.ZoomMustBeInteger(actual_zoom=zoom)
            if not LOWEST_ZOOM <= zoom <= HIGHEST_ZOOM:
                return Status.ERROR, basemap_error.InvalidZoomValue(actual_zoom=zoom)
        return Status.OK, None

    def _check_params(self, requirements: dict, request: dict) -> bool:
        """
        check that the link params correspond to the requirements
        Args:
            requirements: part of the requirements, either recommended or required section
            request: data request, must contain 'url'

        Returns:

        """
        return self._zoom_is_ok(requirements, request)
