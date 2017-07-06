from .base import BaseAPIClient


class TaskClient(BaseAPIClient):
    def get_tasks(self, task_names, limit=1):
        return self.request('get',
                            '/tasks/queue/',
                            params={'tasks': task_names, 'worker_id': self.worker_id, 'limit': limit})

    def release_task(self, task):
        return self.request('post',
                            '/tasks/{id}/release/'.format(id=task['id']),
                            params={'worker_id': self.worker_id})

    def complete_task(self, task):
        return self.request('post',
                            '/tasks/{id}/complete/'.format(id=task['id']),
                            params={'worker_id': self.worker_id})

    def fail_task(self, task):
        return self.request('post',
                            '/tasks/{id}/fail/'.format(id=task['id']),
                            params={'worker_id': self.worker_id})