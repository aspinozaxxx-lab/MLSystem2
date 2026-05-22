from ..base.validator import Validator
from ..base.status import Status


class HeadImageryValidator(Validator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _request_is_ok(self, request: dict):
        return Status.OK, None

    def _check_params(self, rparams: dict, request: dict):
        return Status.OK, None
