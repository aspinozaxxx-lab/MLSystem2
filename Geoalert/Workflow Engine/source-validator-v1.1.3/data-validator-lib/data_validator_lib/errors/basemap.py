from ..base.error_message import ErrorMessage
from ..base.constants import URL_KEY, LOWEST_ZOOM, HIGHEST_ZOOM


class UrlMustBeString(ErrorMessage):
    def __init__(self, url_type):
        super().__init__(message="Parameter \"{URL_KEY}\" must be str, not {{url_type}}".format(URL_KEY=URL_KEY),
                         url_type=url_type)


class UrlMustBeLink(ErrorMessage):
    def __init__(self):
        super().__init__(message="Parameter \"{URL_KEY}\" must be a link "
                                     "starting with http:// or https://".format(URL_KEY=URL_KEY))


class UrlBlacklisted(ErrorMessage):
    def __init__(self, url: str, pattern: str = ""):
        super().__init__(message="Basemap url {url} meets one of blacklisted patterns: " + pattern,
                         url=url)


class UrlFormatError(ErrorMessage):
    def __init__(self, parse_error_message):
        super().__init__(message="{URL_KEY} format is invalid: {{parse_error_message}}".format(URL_KEY=URL_KEY),
                         parse_error_message=parse_error_message)


class ZoomMustBeInteger(ErrorMessage):
    def __init__(self, actual_zoom):
        super().__init__(message="Zoom must be either empty, or integer, got {actual_zoom}",
                         actual_zoom=actual_zoom)


class InvalidZoomValue(ErrorMessage):
    def __init__(self, actual_zoom):
        super().__init__(message="Zoom must be between {LOWEST_ZOOM} and {HIGHEST_ZOOM}, "
                                     "got {{actual_zoom}}".format(LOWEST_ZOOM=LOWEST_ZOOM, HIGHEST_ZOOM=HIGHEST_ZOOM),
                         actual_zoom=actual_zoom)


class TooLowZoom(ErrorMessage):
    def __init__(self, actual_zoom, min_zoom):
        super().__init__(message="Zoom must be higher than {min_zoom}, "
                                     "got {actual_zoom}",
                         actual_zoom=actual_zoom,
                         min_zoom=min_zoom)


class TooHighZoom(ErrorMessage):
    def __init__(self, actual_zoom, max_zoom):
        super().__init__(message="Zoom must be lower than {max_zoom}, "
                                     "got {actual_zoom}",
                         actual_zoom=actual_zoom,
                         max_zoom=max_zoom)