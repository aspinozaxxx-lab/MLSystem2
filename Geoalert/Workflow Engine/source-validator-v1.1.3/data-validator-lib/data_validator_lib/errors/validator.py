from ..base.error_message import ErrorMessage
from typing import List


class RequirementsMustBeDict(ErrorMessage):
    def __init__(self, requirements_type):
        super().__init__(message="Requirements must be dict, got {requirements_type}",
                         requirements_type=requirements_type)


class RequestMustBeDict(ErrorMessage):
    def __init__(self, request_type):
        super().__init__(message="Request must be dict, got {request_type}",
                         request_type=request_type)


class RequestMustHaveSourceType(ErrorMessage):
    def __init__(self):
        super().__init__(message="Request must must contain \"source_type\" key")


class TaskMustContainAoi(ErrorMessage):
    def __init__(self):
        super().__init__(message="Task must must contain area of interest (\"aoi\" key)")


class SourceTypeIsNotAllowed(ErrorMessage):
    def __init__(self, source_type: str, allowed_sources: List[str]):
        super().__init__(message="source type {source_type} is not allowed. Use one of: {allowed_sources}",
                         source_type=source_type,
                         allowed_sources=allowed_sources)


class RequiredSectionMustBeDict(ErrorMessage):
    def __init__(self, required_section_type):
        super().__init__(message="Required section must contain dict, not {required_section_type}",
                         required_section_type=required_section_type)


class RecommendedSectionMustBeDict(ErrorMessage):
    def __init__(self, recommended_section_type):
        super().__init__(message="Recommended section must contain dict, not {recommended_section_type}",
                         recommended_section_type=recommended_section_type)

