import pytest
from data_validator_lib.base.error_message import ErrorMessage
from we_queue_client.message import ErrorMessageSchema


@pytest.mark.parametrize("level", ["error", "warning"])
def test_schema(level):
    error_msg = ErrorMessage(code="test_code",
                             message="Test message: {param1}, {param2}",
                             level=level,
                             param1="value1", param2="value2")
    schema = error_msg.schema()

    assert isinstance(schema, ErrorMessageSchema)
    assert schema.code == "test_code"
    assert schema.parameters == {"param1": "value1",
                                 "param2": "value2"}
    assert schema.message == f"{level.upper()}: Test message: value1, value2"
