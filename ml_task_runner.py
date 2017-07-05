import json
import os
import signal
import sys
import time

import backoff
import nbformat
import nbconvert

from api_clients.tasks import Tasks

class MLTaskRunner(object):
    shutting_down = False
    download_complete = False

    def __init__(self, configuration):
        self.configuration = configuration
        self.tasks_client = Tasks(configuration['services']['task-service']['base_url'],
                                  configuration['auth_token'],
                                  configuration['services']['task-service']['worker_id'])
        self.task = None

    @staticmethod
    def run_notebook(notebook_name, base_path='notebooks/'):
        start_time = time.time()
        print(notebook_name + ' start time: ' + str(start_time))
        output_notebook_filename = notebook_name + '.output.ipynb'
        with open(base_path + notebook_name + '.ipynb') as file:
            notebook = nbformat.read(file, as_version=4)
            preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=-1)
            print('Processing ' + notebook_name + '...')
            preprocessor.preprocess(notebook, {'metadata': {'path': base_path}})
            print(notebook_name + ' processed.')
            with open(base_path + 'output/' + output_notebook_filename, 'wt') as f:
                nbformat.write(notebook, f)
            print(notebook_name + ' output written.')

        end_time = time.time()
        print(notebook_name + ' timing: ' + str(end_time - start_time) + '\n')

    @backoff.on_predicate(backoff.expo, max_value=30, jitter=backoff.full_jitter, factor=2)
    def get_task(self):
        tasks = self.tasks_client.get_tasks(['classifier-search'])

        if len(tasks) > 0:
            return tasks[0]
        else:
            return None

    def run(self):
        while not self.shutting_down:
            self.task = self.get_task()
            print(self.task)

            if self.task is None:
                time.sleep(5)
                continue

            if not self.download_complete:
                self.run_notebook('1.download')

            gene_ids = self.task['data']['genes']
            disease_acronyms = self.task['data']['diseases']

            # os.environ['gene_ids'] = '7157-7158-7159-7161'
            # os.environ['disease_acronyms'] = 'ACC-BLCA'
            os.environ['gene_ids'] = '-'.join(gene_ids)
            os.environ['disease_acronyms'] = '-'.join(disease_acronyms)

            self.run_notebook('2.mutation-classifier')

    def shutdown(self, signum, frame):
        self.shutting_down = True

        # TODO: release task in try/catch/finally

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
