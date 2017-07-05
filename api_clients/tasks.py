from .base import BaseAPIClient

class Tasks(BaseAPIClient):
    def get_tasks(self, task_names, limit=1):
        return self.request('get',
                            '/tasks/queue',
                            params={'tasks': task_names, 'worker_id': self.worker_id, 'limit': limit})
