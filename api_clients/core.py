from .base import BaseAPIClient


class CoreClient(BaseAPIClient):
    def get_classifiers(self, titles, limit=1):
        return self.request('get',
                            '/classifiers/queue/',
                            params={'title': titles, 'worker_id': self.worker_id, 'limit': limit})

    def release_classifier(self, classifier):
        return self.request('post',
                            '/classifiers/{id}/release/'.format(id=classifier['id']),
                            params={'worker_id': self.worker_id})

    def fail_classifier(self, classifier, fail_reason, fail_message):
        return self.request('post',
                            '/classifiers/{id}/fail/'.format(id=classifier['id']),
                            params={'worker_id': self.worker_id},
                            json={
                                'fail_reason': fail_reason,
                                'fail_message': fail_message
                            })

    def upload_notebook(self, classifier, notebook_output_path):
        with open(notebook_output_path, mode='rb') as notebook_file:
            return self.request('post',
                                '/classifiers/{id}/upload/'.format(id=classifier['id']),
                                files={'notebook_file': notebook_file})