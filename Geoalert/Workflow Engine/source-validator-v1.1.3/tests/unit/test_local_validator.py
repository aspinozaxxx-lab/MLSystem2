import pytest
from rasterio import Affine
from data_validator_lib.validators.local import LocalValidator
from data_validator_lib import Status, BadRequirements
from data_validator_lib.errors.local import NChannelsNotAllowed, DtypeNotAllowed

def test_request_is_ok():
    inputs = [
        ({'url': 's3://bwgberbg/feve.tif'}, Status.OK),
        ({'profile': {'crs': '', 'dtype': '', 'transform': (), 'count': 0}}, Status.OK),
        ({'url': 's3://bwgberbg', 'profile': {'crs': '', 'dtype': '', 'transform': (), 'count': 0}}, Status.ERROR),
        ({'url': 's3://bwgberbg/', 'profile': {'crs': '', 'dtype': '', 'transform': (), 'count': 0}}, Status.OK),
        ({'url': 's3://bwgberbg/feve.tif', 'profile': {'crs': '', 'dtype': '', 'transform': (), 'count': 0}}, Status.OK),
        ({'key': 'value'}, Status.ERROR),  # bad keys
        ({'url': 'https://dfvwf/{x}/{y}/{z}.tif'}, Status.ERROR),  # bad link
        ({'profile': {'dtype': '', 'transform': (), 'count': 0}}, Status.ERROR),  # bad profile keys
        ({'profile': {'crs': '', 'transform': (), 'count': 0}}, Status.ERROR),  # bad profile keys
        ({'profile': {'dtype': '', 'crs': '', 'count': 0}}, Status.ERROR),  # bad profile keys
        ({'profile': {'dtype': '', 'transform': (), 'crs': ''}}, Status.ERROR),  # bad profile keys
        # bad profile keys but correct s3 link will be preferrable
        ({'url': 's3://bwgberbg/', 'profile': {'dtype': '', 'transform': (), 'crs': ''}}, Status.OK),
        # good profile keys but error in s3 link will be preferrable
        ({'url': 'MSIL2A_32423452345', 'profile': {'dtype': '', 'transform': (), 'crs': ''}}, Status.ERROR),
    ]

    validator = LocalValidator()
    for request, expected in inputs:
        result = validator._request_is_ok(request)
        assert result[0] == expected, f'{request = }, {result = }, {expected = }'


def test_check_dtype():
    inputs = [
        ('uint8', 'uint8', True),
        (['uint8', 'wevfwe'], 'uint8', True),
        ('uint8', 'uint16', False),
        (['uint8', 'uint168'], 'uint16', False),
        ('uint168', 'uint16', False),  # to test that there is no 'in' condition for strings
    ]
    validator = LocalValidator()
    for required, request, expected in inputs:
        result = validator._check_dtype(required, request)
        assert result == expected,  f'{required = }, {request= }, ' \
                                    f'{result = }, {expected = }'
        if not expected:
            assert isinstance(validator.params_message[0], DtypeNotAllowed)


def test_check_nchannels():
    inputs = [
        (1, 1, True),
        (3, 1, False),
        ([1, 3], 3, True),
        ([1, 10], 3, False),
        (None, 1234, True),
        # non-int args in list are ignored!
        ([1, '2'], 2, False),
        ([1, '2'], 1, True),
    ]
    validator = LocalValidator()
    for required, request, expected in inputs:
        result = validator._check_nchannels(required, request)
        assert result == expected
        if not expected:
            assert isinstance(validator.params_message[0], NChannelsNotAllowed)



def test_check_nchannels_fails():
    validator = LocalValidator()
    with pytest.raises(BadRequirements):
        validator._check_nchannels("1", 1)


def test_check_dtypes_fails():
    """Requirements must have str or list, we must raise exception otherwise"""
    validator = LocalValidator()
    inputs = [
            (12345, 12345, True),
        ]
    for required, request, expected in inputs:
        with pytest.raises(BadRequirements):
            validator._check_dtype(required, request)


def test_check_resolution():
    """ resolution must be calculated as real pixel size at the Earth surface """
    inputs = [
        (1.0, 1.0, 'EPSG:3857', (1.0, 0, 0, 0, -1.0, 0), True), # totally ok
        (1.0, 1.0, 'EPSG:3857', Affine(1.0, 0, 0, 0, -1.0, 0), True),  # also ok - with Affine
        (1.0, 1.0, 'EPSG:3857', (1.02, 0, 0, 0, -1.02, 0), True),  # testing allowed error
        (1, 1, 'EPSG:3857', (0.98, 0, 0, 0, -0.98, 0), True),  # testing allowed error and integer boundary
        (1.1, 1.2, 'EPSG:3857', (1.0, 0, 0, 0, -1.0, 0), False),  # out of res range
        (0.8, 0.9, 'EPSG:3857', (1.0, 0, 0, 0, -1.0, 0), False),  # out of res range
    ]
    validator = LocalValidator()
    for min_res, max_res, crs, transform, expected in inputs:
        result = validator._check_resolution(min_res,
                                             max_res,
                                             crs,
                                             transform)
        assert result == expected,  f'{min_res = }, {max_res= }, ' \
                                    f'{crs = }, {transform = }, ' \
                                    f'{result = }, {expected = } ' + \
                                    validator.params_message['res']


def test_check_resolution_fails():
    validator = LocalValidator()

    min_res, max_res, crs, transform = 1.0, 1.0, 'EPSG:4326', (1.0, 0, 0, 0, -1.0, 0), # out of latitude range
    with pytest.raises(ValueError):
        validator._check_resolution(min_res, max_res, crs, transform)

    min_res, max_res, crs, transform = 1.0, 1.0, 'EPSG:4326', [(1.0, 0, 0, 0, -1.0, 0)],  # bad transform
    with pytest.raises(Exception):
        validator._check_resolution(min_res, max_res, crs, transform)

    min_res, max_res, crs, transform = 1.0, 1.0, 'crs:latlon', [(1.0, 0, 0, 0, -1.0, 0)],  # bad crs
    with pytest.raises(Exception):
        validator._check_resolution(min_res, max_res, crs, transform)


def test_check_params_catches():
    """
    Inside check_params we catch all the errors which arise if the request or requirements are really bad
    In this case, dtype and crs are invalid and cause exceptions
    """

    # bad dtype and bad crs
    validator = LocalValidator()
    requirements = {'dtype': 12345, 'min_res': 1.0, 'max_res': 2.0}
    request = {'profile': {'crs': 'crs:latlon',
                           'dtype': 12345,
                           'transform': (1.0, 0, 0, 0, -1.0, 0),
                           'count': 3}}
    result = validator._check_params(requirements, request)
    assert result is False
    # the exception must be caught and the according description keys must be filled
    assert validator.params_message
    assert validator.params_message
