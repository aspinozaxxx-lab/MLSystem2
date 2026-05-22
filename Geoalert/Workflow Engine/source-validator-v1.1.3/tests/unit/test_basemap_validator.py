import pytest

from data_validator_lib.validators.basemap import BasemapValidator
from data_validator_lib.validators.xyz import XYZValidator
from data_validator_lib.validators.quadkey import QuadkeyValidator
from data_validator_lib.validators.tms import TMSValidator
from data_validator_lib.base.status import Status
from data_validator_lib.base.validator import BadRequirements


def test_basemap_validator_is_abstract():
    """
    BasemapValidator must raise NotImplemented error, use xyz/tms/quadkey class
    """
    request = {'source_type': 'xyz', 'url': 'http://google.com'}
    requirements = {}

    validator = BasemapValidator()

    with pytest.raises(NotImplementedError):
        validator(requirements, request)


def test_raises_runtimeerror():
    """
    Every validator should raise error if called with wrong source_type
    """
    validators = [XYZValidator(), TMSValidator(), QuadkeyValidator()]

    request = {'source_type': 'basemap', 'url': 'http://google.com'}  # source_type does not correspond to any
    requirements = {}

    for validator in validators:
        with pytest.raises(RuntimeError):
            validator(requirements, request)


def test_request_url_is_valid():
    """
    BasemapValidator must return error if url is not present or does not start with http:// or https://
    """
    input_data = [({'url': 'http://google.com'}, Status.OK),
                  ({'url': 'https://google.com'}, Status.OK),
                  ({'url': 'google.com/{x}/{y}/{z}.png'}, Status.ERROR), # no http/https
                  ({'url': {}}, Status.ERROR),# not str
                  ({'key': 'value'}, Status.ERROR),# no url key
                  ({'url': 'https://tile.opentopomap.org/tiles/{x}/{y}/{z}.png'}, Status.ERROR)] # blacklisted
    validator = BasemapValidator()

    for request, result in input_data:
        assert validator._request_url_is_valid(request)[0] == result, str(request)


def test_xyz_validator_request_is_ok():
    """
    XYZ validator requires url with {x} {y} {z} placeholders for format
    """
    input_data = [({'source_type': 'xyz', 'url': 'http://google.com'}, Status.ERROR),  # not {x}{y}{z}
                  ({'source_type': 'xyz', 'url': 'https://google.com/{x}/{y}/{z}.png'}, Status.OK),
                  ({'source_type': 'xyz', 'url': 'http://google.com/z={z}x={x}y={y}'}, Status.OK)]
    validator = XYZValidator()

    for request, result in input_data:
        assert validator._request_is_ok(request)[0] == result, str(request)


def test_tms_validator_request_is_ok():
    """
    TMS validator requires url with {x} {y} {z} placeholders for format
    """
    input_data = [({'source_type': 'tms', 'url': 'http://google.com'}, Status.ERROR),  # not {x}{y}{z}
                  ({'source_type': 'tms', 'url': 'https://google.com/{x}/{y}/{z}.png'}, Status.OK),
                  ({'source_type': 'tms', 'url': 'https://google.com/{x}/{y}/{z}{z}.png'}, Status.ERROR),  # {z} key *2
                  ({'source_type': 'tms', 'url': 'https://google.com/{x}/{y}/{z}{a}.png'}, Status.ERROR),  # {a} key
                  ({'source_type': 'tms', 'url': 'http://google.com/z={z}x={x}y={y}'}, Status.OK)]
    validator = TMSValidator()

    for request, result in input_data:
        assert validator._request_is_ok(request)[0] == result, str(request)


def test_quadkey_validator_request_is_ok():
    """
    Quadkey validator requires url with {q} placeholder for format
    """

    input_data = [({'source_type': 'quadkey', 'url': 'http://google.com/{x}/{y}/{z}'}, Status.ERROR),  # not {q}
                  ({'source_type': 'quadkey', 'url': 'https://google.com/{q}.png'}, Status.OK),
                  ({'source_type': 'quadkey', 'url': 'https://google.com/{q}/{q}.png'}, Status.ERROR), # double {q}
                  ({'source_type': 'quadkey', 'url': 'http://google.com/z={z}q={q}'}, Status.ERROR)]  # {z} placeholder
    validator = QuadkeyValidator()

    for request, result in input_data:
        assert validator._request_is_ok(request)[0] == result, str(request)


def test_zoom_is_ok():
    """
    Zoom must be between min_zoom and max_zoom in case they are specified,
    otherwise boundaries are 0, 21
    """
    validator = BasemapValidator()

    input_data = [# in case of absent min_zoom and/or max_zoom they are taken as [0, 23]
                  ({}, {'zoom': 19}, True),
                  ({}, {'zoom': "19"}, True),
                  ({}, {'zoom': -1}, False),
                  ({}, {'zoom': 33}, False),
                  ({}, {'zoom': 24}, False),
                  ({}, {'zoom': 23}, True),
                  ({}, {'zoom': "whatisit"}, False),

                  ({'min_zoom': -1, 'max_zoom': 33}, {'zoom': -1}, True),
                  # it is invalid, but this function does not check it, as the requiremets are explicit!
                  # See _zoom_is_valid

                  ({'min_zoom': 18, 'max_zoom': 18}, {'zoom': 18}, True),
                  ({'min_zoom': 16, 'max_zoom': 18}, {'zoom': 17}, True),

                  ({'max_zoom': 18}, {'zoom': 18}, True),
                  ({'max_zoom': 18}, {'zoom': 1}, True),
                  ({'max_zoom': 18}, {'zoom': 19}, False),

                  ({'min_zoom': 16}, {'zoom': 18}, True),
                  ({'min_zoom': 16}, {'zoom': 16}, True),
                  ({'min_zoom': 16}, {'zoom': 15}, False)]

    for requirements, request, result in input_data:
        assert validator._zoom_is_ok(requirements, request) == result, str(requirements) + str(request)

    with pytest.raises(BadRequirements):
        validator._zoom_is_ok({'min_zoom': 18, 'max_zoom': 16}, {'zoom': 17})
