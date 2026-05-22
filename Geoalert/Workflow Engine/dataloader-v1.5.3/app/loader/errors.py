class DataloaderError(ValueError):
    """
    this is duplicate of the submodules' error. As we have 3 different submodules, each of them has the same base error.
    The common place is the interface __dict__ which makes dict with error_code, parameters and message
    """
    def __init__(self, message, **parameters):
        self.error_code = self.__class__.__name__
        self.parameters = parameters
        self.parameters = parameters
        try:
            self.message = message.format(**parameters)
        except KeyError:
            # if there are not enough parameters to format the string -
            # it means that there is error in error class definition/call
            raise AssertionError(f"Error classes must provide all keys for message formatting. "
                                 f"The class {self.error_code} with message \"{message}\" "
                                 f"has incompatible parameters: {parameters}")
        super().__init__(self.message)

    def asdict(self):
        return {"code": self.error_code,
                "parameters": {key: str(value) for key, value in self.parameters.items()},
                "message": self.message}

    def __repr__(self):
        return f"{self.error_code}:{self.message}"


class UnknownSourceType(DataloaderError):
    def __init__(self, allowed_source_types, real_source_type):
        super().__init__(message="Source_type must be {allowed_source_types}, got {real_source_type}",
                         allowed_source_types=allowed_source_types,
                         real_source_type=real_source_type)


class MemoryLimitExceeded(DataloaderError):
    def __init__(self, allowed_size, estimated_size):
        super().__init__(message="Estimated RAM usage is {estimated_size} MB, "
                                 "which is more than allowed {allowed_size} MB",
                         allowed_size=allowed_size,
                         estimated_size=estimated_size)


class LoaderArgsError(DataloaderError):
    def __init__(self, argument_name, argument_type, expected_type):
        super().__init__(message="Loader could not cast argument {argument_name} of type {argument_type}"
                                       " to expected type {expected_type}",
                         argument_name=argument_name,
                         argument_type=argument_type,
                         expected_type=expected_type)


class InternalError(DataloaderError):
    def __init__(self, error_message=""):
        super().__init__(message="Internal error in loading data: {error_message}", error_message=error_message)
