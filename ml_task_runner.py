import json
import os
import signal
import sys
import time

import backoff
import nbformat
import nbconvert

from api_clients.task import TaskClient
from api_clients.core import CoreClient

class MLTaskRunner(object):
    shutting_down = False
    download_complete = False

    def __init__(self, configuration):
        self.configuration = configuration
        self.task_client = TaskClient(configuration['services']['task-service']['base_url'],
                                      configuration['auth_token'],
                                      configuration['services']['task-service']['worker_id'])
        self.core_client = CoreClient(configuration['services']['core-service']['base_url'],
                                      configuration['auth_token'],
                                      configuration['services']['task-service']['worker_id'])
        self.task = None

    @staticmethod
    def run_notebook(notebook_name, base_path='notebooks/'):
        notebook_path = os.path.join(os.getcwd(), base_path, notebook_name + '.ipynb')
        output_path = os.path.join(os.getcwd(), base_path, 'output', notebook_name + '.output.ipynb')

        start_time = time.time()
        print(notebook_name + ' start time: ' + str(start_time))
        with open(notebook_path) as file:
            notebook = nbformat.read(file, as_version=4)
            preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=-1)
            print('Processing ' + notebook_name + '...')
            preprocessor.preprocess(notebook, {'metadata': {'path': base_path}})
            print(notebook_name + ' processed.')
            with open(output_path, 'wt') as f:
                nbformat.write(notebook, f)
            print(notebook_name + ' output written.')

        end_time = time.time()
        print(notebook_name + ' timing: ' + str(end_time - start_time) + '\n')
        return output_path

    @backoff.on_predicate(backoff.expo, max_value=30, jitter=backoff.full_jitter, factor=2)
    def get_task(self):
        tasks = self.task_client.get_tasks(['classifier-search'])

        if len(tasks) > 0:
            return tasks[0]
        else:
            return None

    def run(self):
        while not self.shutting_down:
            self.task = self.get_task()

            if self.task is None:
                sleep_time = 5
                print('No task found. Sleeping for {time} seconds...'.format(time=sleep_time))
                time.sleep(sleep_time)
                continue

            print('Starting task {id}: {task}'.format(id=self.task['id'], task=self.task))

            if not self.download_complete:
                self.run_notebook('1.download')
                self.download_complete = True

            gene_ids = self.task['data']['genes']
            disease_acronyms = self.task['data']['diseases']

            # Example:
            # os.environ['gene_ids'] = '7157-7158-7159-7161'
            # os.environ['disease_acronyms'] = 'ACC-BLCA'
            os.environ['gene_ids'] = '-'.join([str(id) for id in gene_ids])
            os.environ['disease_acronyms'] = '-'.join(disease_acronyms)

            try:
                notebook_output_path = self.run_notebook('2.mutation-classifier')
                print('Machine learning completed.')
                print('Uploading notebook to core-service...')
                self.core_client.upload_notebook(self.task, notebook_output_path)

                print('Completing task with task-service...')
                self.task_client.complete_task(self.task)

                print('Task complete.')
            except Exception as error:
                print('Failed to complete task.')
                print(error)
                self.task_client.fail_task(self.task)

    def shutdown(self, signum, frame):
        self.shutting_down = True

        try:
            if self.task is not None:
                self.task_client.release_task(self.task)
                print('Task {id} released.'.format(id=self.task['id']))
            else:
                print('No task to release.')
        except Exception as error:
            print('Encountered error while releasing task {id}.'.format(id=self.task['id']))
            print(error)
        finally:
            print('Shutting down...')
            sys.exit(0)

if __name__ == '__main__':
    filename = os.getenv('COGNOMA_CONFIG', './config/dev.json')

    with open(filename) as config_file:    
        config = json.load(config_file)

    ml_task_runner = MLTaskRunner(config)

    signal.signal(signal.SIGINT, ml_task_runner.shutdown)
    signal.signal(signal.SIGTERM, ml_task_runner.shutdown)

    ml_task_runner.run()
