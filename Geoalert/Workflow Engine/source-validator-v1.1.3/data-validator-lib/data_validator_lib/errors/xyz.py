from ..base.error_message import ErrorMessage
from ..base.constants import URL_KEY


class XYZLinkFormatError(ErrorMessage):
    def __init__(self):
        super().__init__(message="XYZ basemap {URL_KEY} must be a link"
                                     " containing \"x\", \"y\", \"z\" placeholders".format(URL_KEY=URL_KEY))
