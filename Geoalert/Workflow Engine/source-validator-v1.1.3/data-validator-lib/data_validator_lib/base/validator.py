from typing import Tuple, Optional, List
from .error_message import ErrorMessage
from .status import Status
from ..errors import validator as validator_error


class BadRequirements(ValueError):
    """
    This exception is to distinguish the situation when the WD author made a mistake
    and we should report it as an internal error to the API user rather than calling it validation error
    """
    pass


class Validator:
    """
    Base class for every validator, implements __call__ interface
    Validator is abstract because _request_is_ok and _check_params included in __call__ are not implemented
    __call__ returns Status (OK, WARN, ERROR) and a dict with meaningful message that should be passed to caller
    """
    def __init__(self, **kwargs):
        self.params_message: List[ErrorMessage] = []

    @staticmethod
    def _source_type_allowed(requirements: dict, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Checks if the source type is in the list of allowed source type in workflow def
        Returns: Status, description
        """
        if type(requirements) != dict:
            return Status.ERROR, validator_error.RequirementsMustBeDict(type(requirements))
        if type(request) != dict:
            return Status.ERROR, validator_error.RequestMustBeDict(type(request))
        source_type = request.get('source_type', None)
        if source_type is None:
            return Status.ERROR, validator_error.RequestMustHaveSourceType()

        # requirements can be empty at all, which means that any source type and parameters are allowed
        # requirements['sources'] can be empty, meaning the same
        # otherwise, the requested source_type must be listed
        allowed_sources = list((requirements.get('sources', {})).keys())

        if allowed_sources and (source_type not in allowed_sources):
            return Status.ERROR, \
                   validator_error.SourceTypeIsNotAllowed(source_type=source_type, allowed_sources=allowed_sources)
        return Status.OK, None

    def _request_is_ok(self, request: dict) -> Tuple[Status, Optional[ErrorMessage]]:
        """
        Abstract function, needs reimplementation in inherited classes
        Checks if the request itself is valid

        Returns: Status, description
        """
        raise NotImplementedError("request_is_ok is not implemented in Validator class and needs reimplementation "
                                  f"in {type(self).__name__}")

    def _check_params(self, requirements: dict, request: dict) -> bool:
        """
        Abstract function, needs reimplementation in inherited classes
        Checks if parameters in request meet the restrictions in requirements.

        Returns: True if OK, False otherwise
        """
        raise NotImplementedError("request_params_allowed is not implemented in Validator class"
                                  f" and needs reimplementation in {type(self).__name__}")

    def _request_params_allowed(self, requirements: dict, request: dict) -> Tuple[Status, List[ErrorMessage]]:
        """
        Checking the parameters of request if they meet the required and the recommended params from requirements
        Returns ERROR if required params are violated and WARNING if recommended params are violated
        Relies on function _check_params
        Args:
            requirements:
            request:
        Returns:
        """
        if not requirements:
            return Status.OK, []
        # we can be sure that request has 'source_type' key, request has already passed _source_type_allowed check
        request_source_type = request['source_type']
        requirements_sources = requirements.get('sources')

        if not requirements_sources:
            return Status.OK, []

        # get request source type from requirements
        # we can be sure that this section is here because we have checked it in _request_is_ok
        requirements_source_type = requirements_sources[request_source_type]

        if not requirements_source_type:
            return Status.OK, []

        # get required section from requirements_source_type
        required = requirements_source_type.get('required', None)

        # check if request is compatible with required section
        if required:
            self.params_message = []
            if type(required) != dict:
                return Status.ERROR, [validator_error.RequiredSectionMustBeDict(required_section_type=type(required))]
            if not self._check_params(required, request):
                return Status.ERROR, self.params_message

        # get recommended section from requirements_source_type
        recommended = requirements_source_type.get('recommended', None)
        # check if request is compatible with recommended section
        if recommended:
            self.params_message = []
            if type(recommended) != dict:
                return Status.ERROR, [validator_error.RecommendedSectionMustBeDict(recommended_section_type=type(recommended))]
            if not self._check_params(recommended, request):
                # To mark the messages as warnings, not errors:
                for message in self.params_message:
                    message.parameters['level'] = 'warning'
                return Status.WARNING, self.params_message
        return Status.OK, []

    def __call__(self, requirements: dict, request: dict) -> Tuple[Status, List[ErrorMessage]]:
        """
        Main interface of the Validator class, does not need reimplementation
        Sequentially calls _source_type_allowed, _request_is_ok, _request_params_allowed

        This sequention allows us NOT re-check the ERROR-causing
        problems that should have been detected at the previous stage
        If you call the protected function, you may have to check the correctness of input by yourself

        to validate the request against the requirements
        Args:
            requirements: Workflow Definition or part of it, describing the data requirements. See README for more info.
                Must contain 'sources' key
            request: Data request. Must contain source_type key, format of other data depends on the source_type.

        Returns:
            (Status, Description) tuple.
            Status can be 0 (OK), 1 (WARNING) or 2 (ERROR).
            Description is empty in case of OK and describes errors and problems in case of WARNING or ERROR
        """
        status = Status.OK
        # description = {}
        error_messages = []

        source_status, source_description = self._source_type_allowed(requirements=requirements, request=request)
        status = max(status, source_status)
        if source_description:
            # add description only if there is something to add
            # description['source'] = source_description
            error_messages.append(source_description)
        if status == Status.ERROR:
            # we can stop further validation if the current source type is not allowed for these requirements
            return status, error_messages

        request_status, request_description = self._request_is_ok(request=request)
        # we update the status only to increase
        # (it will be OK only if all stages are OK, and ERROR if any stage will give errors)
        status = max(status, request_status)
        if request_description:
            # add description only if there is something to add
            # description['request'] = request_description
            error_messages.append(request_description)

        if status == Status.ERROR:
            # we can stop further validation if the current source type is not allowed for these requirements
            return status, error_messages

        params_status, params_description = self._request_params_allowed(requirements=requirements, request=request)
        status = max(status, params_status)
        if params_description:
            # description['params'] = params_description
            error_messages += params_description
        return status, error_messages
