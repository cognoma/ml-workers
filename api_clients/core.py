from .base import BaseAPIClient


class CoreClient(BaseAPIClient):
    def upload_notebook(self, task, notebook_output_path):
        with open(notebook_output_path, mode='rb') as notebook_file:
            return self.request('post',
                                '/classifiers/{id}/upload/'.format(id=task['id']),
                                files={'notebook_file': notebook_file})
