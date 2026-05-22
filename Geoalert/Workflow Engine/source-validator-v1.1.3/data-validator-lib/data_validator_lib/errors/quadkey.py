from ..base.error_message import ErrorMessage
from ..base.constants import URL_KEY


class QuadkeyLinkFormatError(ErrorMessage):
    def __init__(self):
        super().__init__(message="Quadkey basemap {URL_KEY} must be a link"
                                     " containing \"q\" placeholder".format(URL_KEY=URL_KEY))
