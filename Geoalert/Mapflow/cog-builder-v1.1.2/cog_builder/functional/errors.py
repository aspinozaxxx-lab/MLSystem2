import json

class CogBuilderError(ValueError):
    pass

class InputDataError(ValueError):
    def __init__(self, message):
        super().__init__(f"Provided file can't be processed: {message}")
        self.error_message = message


class CogReprojectionError(CogBuilderError):
    def __init__(self, message):
        super().__init__(f"Provided file cannot be reprojected to WebMercator: {message}")
        self.error_message=message

class CogInvalidAOI(CogBuilderError):
    def __init__(self, aoi, reason: str):
        super().__init__(f"Input AOI {aoi} is invalid because {reason}")
        self.aoi=aoi
        self.reason=reason
