from data_validator_lib.errors.validator import *

def test_requirements_must_be_dict():
    error = RequirementsMustBeDict(requirements_type="str")
    assert error.log_message == "ERROR: Requirements must be dict, got str"

def test_request_must_be_dict():
    error = RequestMustBeDict(request_type="str")
    assert error.log_message == "ERROR: Request must be dict, got str"

def test_request_must_have_source_type():
    error = RequestMustHaveSourceType()
    assert error.log_message == "ERROR: Request must must contain \"source_type\" key"

def test_task_must_contain_aoi():
    error = TaskMustContainAoi()
    assert error.log_message == "ERROR: Task must must contain area of interest (\"aoi\" key)"

def test_source_type_is_not_allowed():
    error = SourceTypeIsNotAllowed(source_type="local", allowed_sources=["xyz", "tms"])
    assert error.log_message == "ERROR: source type local is not allowed. Use one of: ['xyz', 'tms']"

def test_required_section_must_be_dict():
    error = RequiredSectionMustBeDict(required_section_type="str")
    assert error.log_message == "ERROR: Required section must contain dict, not str"

def test_recommended_section_must_be_dict():
    error = RecommendedSectionMustBeDict(recommended_section_type=type("String Type"))
    assert error.log_message == "ERROR: Recommended section must contain dict, not <class 'str'>"
