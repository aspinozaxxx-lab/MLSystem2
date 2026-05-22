from typing import Optional
from we_queue_client.message import ErrorMessageSchema


class ValidationErrorMessageSchema(ErrorMessageSchema):
    level: str = 'error'


class BadMessageParameters(ValueError):
    pass

class ErrorMessage:
    """
    Base class for all error messages. It is not an exceeption because we want to accumulate
    multiple errors in one validation run
    """
    def __init__(self, code: Optional[str] = None, message: str = "", level: str = 'error', **parameters):
        self.code = code or type(self).__name__
        self.parameters = {key: str(value) for key, value in parameters.items()}
        self.parameters['level'] = level
        try:
            self._log_message = message.format(**self.parameters)
        except KeyError as e:
            raise BadMessageParameters(f"Message {message} format must match the parameters."
                                       f" Parameter {e} is not provided")

    def __repr__(self):
        return f"Data Validator: code = {self.code}, log = {self.log_message}, params = {self.parameters}"

    @property
    def log_message(self):
        return self.parameters['level'].upper() + ": " + self._log_message

    def schema(self):
        """
        Convert to ErrorMessageSchema for sending to queue
        """
        return ValidationErrorMessageSchema(code=self.code,
                                            parameters={k:v for k,v in self.parameters.items() if k != 'level'},
                                            level=self.parameters['level'],
                                            message=self.log_message)


class OK(ErrorMessage):
    # do we really need it? Will use None by now
    def __init__(self):
        super().__init__(code="OK", message="OK")


class InternalError(ErrorMessage):
    def __init__(self, exception):
        super().__init__(code="InternalError",
                         message="Internal error: " + str(exception))
