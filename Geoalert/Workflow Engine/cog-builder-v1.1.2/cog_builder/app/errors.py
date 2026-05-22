from we_queue_client import QueueWorkerError


class ReprojectionError(QueueWorkerError):
    def __init__(self, error_message):
        super().__init__(message=f"Provided file cannot be reprojected to WebMercator: {error_message}")


class InvalidAOI(QueueWorkerError):
    def __init__(self, aoi, reason: str):
        super().__init__(message="Input AOI {aoi} is invalid because {reason}",
                         aoi=aoi,
                         reason=reason)
