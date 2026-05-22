from pydantic import ValidationError
from we_queue_client.errors import QueueWorkerError


class InvalidAOI(QueueWorkerError):
    def __init__(self, aoi, reason: str):
        super().__init__(message="Input AOI is invalid because {reason}",
                         aoi=aoi,
                         reason=reason)

class ConfigValidationError(QueueWorkerError):
    def __init__(self, message, brick_class, **errors):
        super().__init__(message, brick_class=brick_class, **errors)

    @classmethod
    def from_pydantic_error(cls, e: ValidationError):
        errors = e.errors()
        out_errors = {}
        message = "Validation failed for {brick_class}."
        for i, err in enumerate(errors):
            field_key = f"field{i}"
            type_key = f"type{i}"

            field_val = err.get('loc')
            if len(field_val) == 1:
                # single-value loc means that the error is not nested, so no need to use the tuple
                field_val = field_val[0]
            type_val = err.get('type')

            message += (" Field: {{{field_key}}}, Error: {{{type_key}}}.".format(field_key=field_key, type_key=type_key))
            out_errors[field_key] = field_val
            out_errors[type_key] = type_val

        return cls(message=message, brick_class = e.title, **out_errors)