from ..base.error_message import ErrorMessage
from ..base.constants import INPUT_STRING_KEY


class SentinelInputStringKeyError(ErrorMessage):
    def __init__(self):
        super().__init__(message=f"Sentinel_L2A request must contain field "
                                     f"named `{INPUT_STRING_KEY}` with string value")


class SentinelInputStringFormatError(ErrorMessage):
    def __init__(self, input_string):
        super().__init__(message=f"Input string `{input_string}` is of unknown format",
                         input_string=input_string)


class GridCellOutOfBounds(ErrorMessage):
    def __init__(self, actual_cell, allowed_cells):
        super().__init__(message="Got grid cell {actual_cell}, requirements are: {allowed_cells}",
                         actual_cell=actual_cell,
                         allowed_cells=allowed_cells)


class AOINotInCell(ErrorMessage):
    def __init__(self, actual_cell):
        super().__init__(message="AOI does not intersect the selected Sentinel-2 granule {actual_cell}",
                         actual_cell=actual_cell)


class MonthOutOfBounds(ErrorMessage):
    def __init__(self, actual_month, allowed_months):
        super().__init__(message="Got image from month {actual_month}, requirements are: {allowed_months}",
                         actual_month=actual_month,
                         allowed_months=allowed_months)

