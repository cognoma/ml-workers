import backoff
import json
import os

from api_clients.tasks import Tasks
from api_clients.cognoma import Cognoma

#from cognoml.analysis import classify

class MLTaskRunner(object):
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
        while True:
            task = self.get_task()

            print(task)

            ## TODO: use task['data'] to calculate sample_id and mutation_status pseudo table

if __name__ == '__main__':
    filename = os.getenv('COGNOMA_CONFIG', './config/dev.json')

    with open(filename) as config_file:    
        config = json.load(config_file)

    ml_task_runner = MLTaskRunner(config)

    ml_task_runner.run()
