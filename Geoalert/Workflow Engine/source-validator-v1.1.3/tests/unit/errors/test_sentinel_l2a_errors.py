from data_validator_lib.errors.sentinel_l2a import *

def test_sentinel_input_string_key_error():
    error_message = SentinelInputStringKeyError()
    assert error_message.log_message == "ERROR: Sentinel_L2A request must contain field named `url` with string value"

def test_sentinel_input_string_format_error():
    error_message = SentinelInputStringFormatError(input_string="test")
    assert error_message.log_message == "ERROR: Input string `test` is of unknown format"

def test_grid_cell_out_of_bounds():
    error_message = GridCellOutOfBounds(actual_cell="38tvu", allowed_cells=["40ttn"])
    assert error_message.log_message == "ERROR: Got grid cell 38tvu, requirements are: ['40ttn']"

def test_aoi_not_in_cell():
    error_message = AOINotInCell(actual_cell="38tvu")
    assert error_message.log_message == "ERROR: AOI does not intersect the selected Sentinel-2 granule 38tvu"

def test_month_out_of_bounds():
    error_message = MonthOutOfBounds(actual_month="10", allowed_months=["1"])
    assert error_message.log_message == "ERROR: Got image from month 10, requirements are: ['1']"
