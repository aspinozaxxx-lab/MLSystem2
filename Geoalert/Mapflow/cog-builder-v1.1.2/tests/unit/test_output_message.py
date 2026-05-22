from we_queue_client.message import TaskStatus
from cog_builder.app.errors import ReprojectionError
from cog_builder.app.message import CogBuilderOutputMessage


def test_success_message():
    assert CogBuilderOutputMessage(task_id=42,
                         status=TaskStatus.OK,
                         error_messages=[]).dict() == {'task_id': 42, 'status': 0, 'messages': []}


def test_error_message():
    expected = {'task_id': 42,
                'status': 1,
                'messages': [{'code': 'ReprojectionError',
                              'parameters': {},
                              'message': 'Provided file cannot be reprojected to WebMercator: Bad CRS'}]}

    error_messages = [ReprojectionError(error_message='Bad CRS').schema()]
    assert CogBuilderOutputMessage(task_id=42,
                         status=TaskStatus.FAIL,
                         messages=error_messages).dict() == expected
