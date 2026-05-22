import re
import inspect
from pathlib import Path
from typing import Tuple, List, Optional
from ..functional.geometry import aoi_intersects_cell
from ..base.status import Status
from ..base.validator import Validator, BadRequirements
from ..base.error_message import ErrorMessage
from ..base.constants import INPUT_STRING_KEY, AOI_KEY
from ..errors import sentinel_l2a as sentinel_error
from ..errors.validator import TaskMustContainAoi

# config - constants
PRODUCT_NAME_COORDS_PATTERN = r"\/(\d{2})\/([A-Z])\/([A-Z]{2})\/"
PRODUCT_NAME_DATETIME_PATTERN = r"\/(20[0-2][0-9])\/(1[0-2]|0?[1-9])\/(0?[1-9]|[1-2]\d|3[0-1])\/"
PRODUCT_NAME_SEQUENCE_PATTERN = r"\/(?:20[0-2][0-9])\/(?:1[0-2]|0?[1-9])\/(?:0?[1-9]|[1-2]\d|3[0-1])\/(\d{1,2})\/$"

SENTINEL_ID_COORDS_PATTERN = r"[T,t](\d{2}[A-Za-z]{3})"
SENTINEL_ID_DATETIME_PATTERN = r"_(\d{4})(\d{2})(\d{2})T(\d{6})"
###

# path to local file with grid, that is initialized in build time, not meant to be changed in runtime


class SentinelL2AValidator(Validator):
    def __init__(self, **kwargs):
        # the file is a part of the package and its relative path is fixed here
        self.metadata_csv_path = str(Path(inspect.getfile(inspect.currentframe())).parent.parent/'static'/'mgrs_grid.csv')
        super().__init__(**kwargs)

        self.product_name_coords_pattern = re.compile(PRODUCT_NAME_COORDS_PATTERN)
        self.product_name_datetime_pattern = re.compile(PRODUCT_NAME_DATETIME_PATTERN)
        self.product_name_sequence_pattern = re.compile(PRODUCT_NAME_SEQUENCE_PATTERN)
        self.product_name_patterns = (
            self.product_name_coords_pattern,
            self.product_name_datetime_pattern,
            self.product_name_sequence_pattern,
        )

        self.sentinel_id_coords_pattern = re.compile(SENTINEL_ID_COORDS_PATTERN)
        self.sentinel_id_datetime_pattern = re.compile(SENTINEL_ID_DATETIME_PATTERN)
        self.sentinel_id_patterns = (
            self.sentinel_id_coords_pattern,
            self.sentinel_id_datetime_pattern,
        )

    def _is_product_name(self, input_string: str) -> bool:
        return all((bool(pattern.search(input_string)) for pattern in self.product_name_patterns))

    def _is_sentinel_id(self, input_string: str) -> bool:
        return all((pattern.search(input_string) for pattern in self.sentinel_id_patterns))

    def _get_cell(self, input_string: str) -> str:
        if self._is_sentinel_id(input_string=input_string):
            cell = self.sentinel_id_coords_pattern.search(input_string)[1]
        elif self._is_product_name(input_string=input_string):
            match = self.product_name_coords_pattern.search(input_string)
            cell = "".join(match.groups())
        else:
            # actually, this must be unreachable code
            raise RuntimeError('sentinel_l2a input string must have been checked for format earlier!')
        return cell

    def _check_cells(self, allowed_cells: List[str], input_string: str) -> bool:
        """
        checks if the MGRS grid cell in input_string of sentinel_l2a dataloader is allowed
        sets params_message['cells'] message
        Args:
            allowed_cells: list of cells parsed from requirements
            input_string: sentinel-l2a loader input string
        Returns:
            true if cell is allowed, false otherwise
        """
        cell = self._get_cell(input_string)
        if cell in allowed_cells:
            return True
        self.params_message.append(sentinel_error.GridCellOutOfBounds(actual_cell=cell, allowed_cells=allowed_cells))
        return False

    def _check_months(self, allowed_months: List[int], input_string: str) -> bool:
        """
        checks if the month in input_string of sentinel_l2a dataloader is allowed
        sets params_message['months'] message
        Args:
            allowed_cells: list of cells parsed from requirements
            input_string: sentinel-l2a loader input string
        Returns:
            true if cell is allowed, false otherwise
        """
        if self._is_sentinel_id(input_string):
            month = self.sentinel_id_datetime_pattern.search(input_string)[2]
        elif self._is_product_name(input_string):
            month = self.product_name_datetime_pattern.search(input_string)[2]
        else:
            # actually, this must be unreachable code
            raise RuntimeError('sentinel_l2a input string must have been checked for format earlier!')
        if int(month) in allowed_months:
            return True
        self.params_message.append(sentinel_error.MonthOutOfBounds(actual_month=month, allowed_months=allowed_months))
        return False
    # ============= implementation of Validator abstract functions ============ #

    def _request_is_ok(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        input_sting = request.get(INPUT_STRING_KEY, None)
        if type(input_sting) != str:
            return Status.ERROR, sentinel_error.SentinelInputStringKeyError()
        input_string = request[INPUT_STRING_KEY]
        if not (self._is_sentinel_id(input_string) or self._is_product_name(input_string)):
            return Status.ERROR, sentinel_error.SentinelInputStringFormatError(input_string)
        aoi = request.get(AOI_KEY, None)
        if not aoi:
            return Status.ERROR, TaskMustContainAoi()
        cell = self._get_cell(input_string)
        if not aoi_intersects_cell(cell=cell, aoi=aoi, metadata_csv_path=self.metadata_csv_path):
            return Status.ERROR, sentinel_error.AOINotInCell(actual_cell=cell)

        return Status.OK, None

    def _check_params(self, params: dict, request: dict) -> bool:
        input_string = request.get(INPUT_STRING_KEY, None)
        if not input_string:
            raise RuntimeError('sentinel_l2a input string existence must have been checked earlier!')

        cells = params.get('cells', None)
        if cells and not self._check_cells(cells, input_string):
            return False

        months = params.get('months', None)
        if months:
            try:
                int_months = [int(entry) for entry in months]
            except Exception:
                raise BadRequirements(f'Requirements[months] must be a list of ints (or allow casting), gon {months}.')
            if not self._check_months(int_months, input_string):
                return False
        return True


