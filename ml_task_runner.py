import os
import signal
import sys
import time
from pathlib import Path

import backoff
import nbformat
import nbconvert
import settings

from api_clients.core import CoreClient

def run_notebook(notebook_name, notebooks_folder_name='notebooks'):
        notebook_path = Path('.', notebooks_folder_name, notebook_name + '.ipynb')
        output_folder_path = Path('.', notebooks_folder_name, 'output')
        output_notebook_path = Path(output_folder_path, notebook_name + '.output.ipynb')

        start_time = time.perf_counter()
        print(notebook_name + ' start time: ' + str(start_time))
        with notebook_path.open() as notebook_file:
            notebook = nbformat.read(notebook_file, as_version=4)
            preprocessor = nbconvert.preprocessors.ExecutePreprocessor(timeout=-1)
            print('Processing ' + notebook_name + '...')
            preprocessor.preprocess(notebook, {'metadata': {'path': notebooks_folder_name}})
            print(notebook_name + ' processed.')

            if not output_folder_path.is_dir():
                output_folder_path.mkdir()

            with output_notebook_path.open('wt') as f:
                nbformat.write(notebook, f)
            print(notebook_name + ' output written.')

        end_time = time.perf_counter()
        print(notebook_name + ' timing: ' + str(end_time - start_time) + '\n')
        return str(output_notebook_path.resolve())

class MLTaskRunner(object):
    shutting_down = False
    download_complete = False

    def __init__(self):
        self.core_client = CoreClient(settings.base_url,
                                      settings.auth_token,
                                      settings.worker_id)
        self.classifier = None

    @backoff.on_predicate(backoff.expo, max_value=30, jitter=backoff.full_jitter, factor=2)
    def get_classifier(self):
        classifiers = self.core_client.get_classifiers(['classifier-search'])

        if len(classifiers) > 0:
            return classifiers[0]
        else:
            return None

    def run(self):
        while not self.shutting_down:
            try:
                if not self.download_complete:
                    run_notebook('1.download')
                    self.download_complete = True
            except Exception as error:
                print('Failed to run download notebook.')
                print(error)
                os.kill(os.getpid(), signal.SIGTERM)

            self.classifier = self.get_classifier()

            if self.classifier is None:
                sleep_time = 5
                print('No classifier found. Sleeping for {time} seconds...'.format(time=sleep_time))
                time.sleep(sleep_time)
                continue

            print('Starting classifier {id}: {classifier}'.format(id=self.classifier['id'], classifier=self.classifier))

            gene_ids = self.classifier['genes']
            disease_acronyms = self.classifier['diseases']

            # Example:
            # os.environ['gene_ids'] = '7157-7158-7159-7161'
            # os.environ['disease_acronyms'] = 'ACC-BLCA'
            os.environ['gene_ids'] = '-'.join(str(id) for id in gene_ids)
            os.environ['disease_acronyms'] = '-'.join(disease_acronyms)

            try:
                notebook_output_path = run_notebook('2.mutation-classifier')
                print('Machine learning completed.')
                print('Uploading notebook to core-service...')
                self.core_client.upload_notebook(self.classifier, notebook_output_path)

                print('Task complete.')
            except Exception as error:
                print('Failed to complete classifier.')
                print(error)
                self.core_client.fail_classifier(self.classifier)

    def shutdown(self, signum, frame):
        self.shutting_down = True

        try:
            if self.classifier is not None:
                self.core_client.release_classifier(self.classifier)
                print('Task {id} released.'.format(id=self.classifier['id']))
            else:
                print('No classifier to release.')
        except Exception as error:
            print('Encountered error while releasing classifier {id}.'.format(id=self.classifier['id']))
            print(error)
        finally:
            print('Shutting down...')
            sys.exit(0)

if __name__ == '__main__':
    ml_classifier_runner = MLTaskRunner()

    signal.signal(signal.SIGINT, ml_classifier_runner.shutdown)
    signal.signal(signal.SIGTERM, ml_classifier_runner.shutdown)

    ml_classifier_runner.run()
