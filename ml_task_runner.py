import json
import os
import signal
import sys
import time

import backoff
import nbformat
import nbconvert

from api_clients.tasks import Tasks
from api_clients.cognoma import Cognoma

cognoma_machine_learning_dir = 'cognoma-machine-learning/'

class MLTaskRunner(object):
    shutting_down = False
    download_complete = False

    def __init__(self, config):
        self.config = config

    #     self.tasks = Tasks(config['services']['task-service']['base_url'],
    #                        config['auth_token'])
    #
    #     self.core = Cognoma(config['services']['core-service']['base_url'],
    #                         config['auth_token'])
    #
    # @backoff.on_predicate(backoff.expo,
    #                       max_value=30,
    #                       jitter=backoff.full_jitter,
    #                       factor=2)

    def get_task(self):
        tasks = self.tasks.get_tasks(['classifier-search'])

        if len(tasks) > 0:
            return tasks[0]
        else:
            return False

    def run_notebook(self, notebook_name, base_path=cognoma_machine_learning_dir):
        start_time = time.time()
        print(notebook_name + ' start time: ' + str(start_time))
        output_notebook_filename = notebook_name + '.output.ipynb'
        with open(base_path + notebook_name + '.ipynb') as file:
            notebook = nbformat.read(file, as_version=4)
            preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=-1)
            print('Processing ' + notebook_name + '...')
            preprocessor.preprocess(notebook, {'metadata': {'path': base_path}})
            print(notebook_name + ' processed.')
            with open('output/' + output_notebook_filename, 'wt') as f:
                nbformat.write(notebook, f)
            print(notebook_name + ' output written.')

        end_time = time.time()
        print(notebook_name + ' timing: ' + str(end_time - start_time) + '\n')

    def run(self):
        while not self.shutting_down:
            # self.task = self.get_task()
            # print(self.task)

            if not self.download_complete:
                self.run_notebook('1.download')

            # TODO: surround task in try/catch block
            # TODO: use task['data'] to calculate sample_id and mutation_status pseudo table

            notebook_filename = '2.mutation-classifier'
            self.run_notebook(notebook_filename)

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
