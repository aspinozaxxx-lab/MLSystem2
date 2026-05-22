from ..config import config


class OutputMessage:
    SUCCESS = 0
    FAIL = 1

    def __init__(self, task_id, status, error_messages=None):
        self.task_id = task_id
        self.status = status
        self.error_messages = error_messages or []
        for message in error_messages:
            message['code'] = '.'.join((config.WORKER_NAME, message['code']))

    def get(self):
        return {
            'task_id': self.task_id,
            'status': self.status,
            'messages':  self.error_messages
        }
