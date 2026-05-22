from fastapi import status
from config import Config


class DataCatalogError(Exception):
    def __init__(self, message, http_code: status = status.HTTP_500_INTERNAL_SERVER_ERROR, **parameters):
        self.error_code = self.__class__.__name__
        self.parameters = parameters
        self._http_code = http_code
        try:
            self.message = message.format(**parameters)
        except KeyError:
            # if there are not enough parameters to format the string -
            # it means that there is error in error class definition/call
            raise AssertionError(f"Error classes must provide all keys for message formatting. "
                                 f"The class {self.error_code} with message \"{message}\" "
                                 f"has incompatible parameters: {parameters}")
        super().__init__(self.message)

    def detail(self, message=False):
        if message:
            return {
                "code": '.'.join((Config.SERVICE_NAME, self.error_code)),
                "parameters": self.parameters,
                "message": self.message,
                "http_code": self.http_code
            }
        return {
            "code": self.error_code,
            "parameters": self.parameters,
        }

    @property
    def http_code(self):
        return self._http_code

    def __repr__(self):
        return f'{self.error_code}:{self.message}'

    def __str__(self):
        return self.__repr__()
