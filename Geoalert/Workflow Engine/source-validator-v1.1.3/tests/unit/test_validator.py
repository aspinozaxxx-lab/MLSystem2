import pytest


from data_validator_lib.base.validator import Validator
from data_validator_lib.base.status import Status


def test_allowed_sources_ok_for_empty_wd():
    validator = Validator()
    wds = [{},
           {'version': '1.0', 'sources': {}}]
    requests = [{'source_type': 'xyz'},
                {'source_type': 'alkdjfvlskjfevb'}]

    for wd in wds:
        for request in requests:
            assert validator._source_type_allowed(wd, request) == (Status.OK, None)


def test_allowed_sources_fail():
    validator = Validator()
    wd = {'version': '1.0', 'sources': {'sentinel_l2a': None, 'local': {'nchannels': 3}}}
    requests = [{'source_type': 'xyz'},  # wrong source
                {'some_key': 'some_value'}]  # invalid source
    for request in requests:
        assert validator._source_type_allowed(wd, request)[0] == Status.ERROR


def test_allowed_sources_ok():
    validator = Validator()
    wd = {'version': '1.0', 'sources': {'sentinel_l2a': None, 'local': {'nchannels': 3}}}
    requests = [{'source_type': 'sentinel_l2a'},
                {'source_type': 'local'}]
    for request in requests:
        assert validator._source_type_allowed(wd, request) == (Status.OK, None)
