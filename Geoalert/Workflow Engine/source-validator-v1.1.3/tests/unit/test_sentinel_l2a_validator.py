import pytest

from data_validator_lib.validators.sentinel_l2a import SentinelL2AValidator
from data_validator_lib.base.status import Status
from data_validator_lib.base.validator import BadRequirements


def test_request_is_ok():
    """
    Validation of the request: it must return ERROR on invalid or absent 'url' and OK on 'url' correct for dataloader
    """
    examples = [({'source_type': 'sentinel'}, Status.ERROR),  #wrong name
                ({'source_type': 'sentinel_l2a'}, Status.ERROR),  # empty url
                ({'source_type': 'sentinel_l2a', 'url': 'L1C_T37EA_A022398_2021020T084436'}, Status.ERROR),  # wrong url
                ({'source_type': 'sentinel_l2a', 'url': 'L1C_T37UEA_A022398_20210620T084436'}, Status.ERROR),  # no aoi
                ({'source_type': 'sentinel_l2a',
                  'url': 'L1C_T38VNP_A022398_20210620T084436',
                  'aoi': {"type": "Polygon",
                          "coordinates": [[[45.722466910376255, 61.52350527049347],
                                           [45.74091348225352, 61.52350527049347],
                                           [45.74091348225352, 61.514322015710434],
                                           [45.722466910376255, 61.514322015710434],
                                           [45.722466910376255, 61.52350527049347]]]
                          }}, Status.OK),  # ok
                ({'source_type': 'sentinel_l2a',
                  'url': '/38/V/NP/2018/9/29/0/',
                  'aoi': {"type": "Polygon",
                          "coordinates": [[[45.722466910376255, 61.52350527049347],
                                           [45.74091348225352, 61.52350527049347],
                                           [45.74091348225352, 61.514322015710434],
                                           [45.722466910376255, 61.514322015710434],
                                           [45.722466910376255, 61.52350527049347]]]
                          }
                  }, Status.OK)
                ]
    validator = SentinelL2AValidator()
    for request, result in examples:
        assert validator._request_is_ok(request)[0] == result, f'{result} expected for {request}'


def test_check_months():
    """
    Checking the params: if requirements specify the months that are OK for processing
    """
    examples = [([6], 'L1C_T37UEA_A022398_20210620T084436', True),  # ok
                ([8, 9, 10], 'L1C_T37UEA_A022398_20210620T084436', False),  # wrong month
                ([8, 9, 10], '/37/U/CS/2018/9/29/0/', True),  # ok
                ([6], '/37/U/CS/2018/9/29/0/', False)  # wrong month
                ]
    validator = SentinelL2AValidator()
    for months, request, result in examples:
        assert validator._check_months(months, request) == result, f'{result} expected for {request}, {months}'


def test_check_cells():
    """
    Checking the params: if requirements specify the MGRS cells that are OK for processing
    """
    examples = [(['37UEA', 'wefvw'], 'L1C_T37UEA_A022398_20210620T084436', True),  # ok
                (['37UCS', ], 'L1C_T37UEA_A022398_20210620T084436', False),  # wrong month
                (['37UCS'], '/37/U/CS/2018/9/29/0/', True),  # ok
                (['37UEA', 'wefvw'], '/37/U/CS/2018/9/29/0/', False)  # wrong month
                ]
    validator = SentinelL2AValidator()
    for cells, request, result in examples:
        assert validator._check_cells(cells, request) == result, f'{result} expected for {request}, {cells}'


def test_check_params():
    """
    Checking the params as a whole, mainly to test request preparation for the check_cells and check_months
    """
    examples = [({}, {'url': 'L1C_T37UEA_A022398_20210620T084436'}, True),  # empty
                ({'cells': ['37UEA'], 'months':[1, 2, 3, 4, 5, 6]}, {'url': 'L1C_T37UEA_A022398_20210620T084436'}, True),
                ({'cells': ['37UEA']}, {'url': 'L1C_T37UEA_A022398_20210620T084436'}, True),
                ({'months':[1, 2, 3, 4, 5, 6]}, {'url': 'L1C_T37UEA_A022398_20210620T084436'}, True),

                ({}, {'url': '/37/U/CS/2018/9/29/0/'}, True),  # empty
                ({'cells': ['37UCS'], 'months': [1, 2, 3, 4, 5, 9]}, {'url': '/37/U/CS/2018/9/29/0/'}, True),
                ({'cells': ['37UCS']}, {'url': '/37/U/CS/2018/9/29/0/'}, True),
                ({'months': [1, 2, 3, 4, 5, 9]}, {'url': '/37/U/CS/2018/9/29/0/'}, True),

                ({'cells': ['37UEA'], 'months': [1, 2, 3, 4, 5, 6]}, {'url': '/37/U/CS/2018/9/29/0/'}, False),
                ({'cells': ['37UEA']}, {'url': '/37/U/CS/2018/9/29/0/'}, False),
                ({'months': [1, 2, 3, 4, 5, 6]}, {'url': '/37/U/CS/2018/9/29/0/'}, False)
                ]
    validator = SentinelL2AValidator()
    for requirements, request, result in examples:
        assert validator._check_params(requirements, request) == result, f'{result} expected for {request}, {requirements}'


def test_check_params_raises():
    """
    The check should raise internal error if the WD is malformed
    """
    validator = SentinelL2AValidator()
    with pytest.raises(BadRequirements):
        validator._check_params({'months': [1, '2', 'deswfewf']},
                        {'url': '/37/U/CS/2018/9/29/0/', 'source_type': 'sentinel_l2a'})

    with pytest.raises(BadRequirements):
        validator._check_params({'months': 'January'},
                        {'url': '/37/U/CS/2018/9/29/0/', 'source_type': 'sentinel_l2a'})
