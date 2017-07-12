from .base import BaseAPIClient


class CoreClient(BaseAPIClient):
    def get_classifiers(self, titles, limit=1):
        return self.request('get',
                            '/classifiers/queue/',
                            params={'title': titles, 'worker_id': self.worker_id, 'limit': limit})

    def release_classifier(self, task):
        return self.request('post',
                            '/classifiers/{id}/release/'.format(id=task['id']),
                            params={'worker_id': self.worker_id})

    def fail_classifier(self, task):
        return self.request('post',
                            '/classifiers/{id}/fail/'.format(id=task['id']),
                            params={'worker_id': self.worker_id})

    def upload_notebook(self, task, notebook_output_path):
        with open(notebook_output_path, mode='rb') as notebook_file:
            return self.request('post',
                                '/classifiers/{id}/upload/'.format(id=task['id']),
                                files={'notebook_file': notebook_file})