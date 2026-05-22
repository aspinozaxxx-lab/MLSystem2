from data_validator_lib.errors.basemap import *
from data_validator_lib.errors.tms import TMSLinkFormatError
from data_validator_lib.errors.xyz import XYZLinkFormatError
from data_validator_lib.errors.quadkey import QuadkeyLinkFormatError

def test_quadkey_link_format_error_message():
    error_message = QuadkeyLinkFormatError()
    assert error_message.log_message == "ERROR: Quadkey basemap url must be a link containing \"q\" placeholder"

def test_tms_link_format_error():
    error_message = TMSLinkFormatError()
    assert error_message.log_message == "ERROR: TMS basemap url must be a link containing  \"x\", \"y\", \"z\" placeholders"

def test_xyz_link_format_error():
    error_message = TMSLinkFormatError()
    assert error_message.log_message == "ERROR: TMS basemap url must be a link containing  \"x\", \"y\", \"z\" placeholders"

def test_url_must_be_string_error_message():
    error_message = UrlMustBeString(url_type=type(10))
    assert error_message.log_message == "ERROR: Parameter \"url\" must be str, not <class 'int'>"


def test_url_must_be_link_error_message():
    error_message = UrlMustBeLink()
    assert error_message.log_message == "ERROR: Parameter \"url\" must be a link starting with http:// or https://"


def test_url_blacklisted_error_message():
    error_message = UrlBlacklisted(url="http://example.com", pattern="example")
    assert error_message.log_message == "ERROR: Basemap url http://example.com meets one of blacklisted patterns: example"


def test_url_format_error_message():
    error_message = UrlFormatError(parse_error_message="invalid syntax")
    assert error_message.log_message == "ERROR: url format is invalid: invalid syntax"


def test_zoom_must_be_integer_error_message():
    error_message = ZoomMustBeInteger(actual_zoom="10.5")
    assert error_message.log_message == "ERROR: Zoom must be either empty, or integer, got 10.5"


def test_invalid_zoom_value_error_message():
    error_message = InvalidZoomValue(actual_zoom="50")
    assert error_message.log_message == "ERROR: Zoom must be between 0 and 23, got 50"


def test_too_low_zoom_error_message():
    error_message = TooLowZoom(actual_zoom="2", min_zoom="5")
    assert error_message.log_message == "ERROR: Zoom must be higher than 5, got 2"


def test_too_high_zoom_error_message():
    error_message = TooHighZoom(actual_zoom="20", max_zoom="18")
    assert error_message.log_message == "ERROR: Zoom must be lower than 18, got 20"
