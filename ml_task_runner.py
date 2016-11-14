import json
import os
import signal
import sys
import time

import backoff

from api_clients.tasks import Tasks
from api_clients.cognoma import Cognoma

from cognoml.analysis import CognomlClassifier
from cognoml.data import CognomlData

class MLTaskRunner(object):
    shuttingdown = False

    def __init__(self, config):
        self.config = config

        self.tasks = Tasks(config['services']['task-service']['base_url'],
                           config['auth_token'])

        self.core = Cognoma(config['services']['core-service']['base_url'],
                            config['auth_token'])

    @backoff.on_predicate(backoff.expo,
                          max_value=30,
                          jitter=backoff.full_jitter,
                          factor=2)
    def get_task(self):
        tasks = self.tasks.get_tasks(['classifier-search'])

        if len(tasks) > 0:
            return tasks[0]
        else:
            return False

    def run(self):
        while self.shuttingdown == False:
            # self.task = self.get_task()
            # print(self.task)

            ## TODO: surround task in try/catch block

            start_time = time.time()

            a = CognomlData(mutations_json_url='https://github.com/cognoma/machine-learning/raw/876b8131bab46878cb49ae7243e459ec0acd2b47/data/api/hippo-input.json',
                            directory='/data/downloads')
            x, y = a.run()

            ## TODO: use task['data'] to calculate sample_id and mutation_status pseudo table

            classifier = CognomlClassifier(x, y)
            classifier.fit()
            results = classifier.get_results()

            end_time = time.time()

            json_results = json.dumps(results, indent=2)
            print(json_results)

            print('Timing ' + str(end_time - start_time))

    def shutdown(self, signum, frame):
        self.shuttingdown = True

        ## TODO: release task in try/catch/finally

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
