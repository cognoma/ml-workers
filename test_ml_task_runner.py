import os
from ml_task_runner import run_notebook

print('Testing download notebook.')
run_notebook('1.download')

os.environ['gene_ids'] = '7157-7158-7159-7161'
os.environ['disease_acronyms'] = 'ACC-BLCA'

print('Testing classifier notebook.')
try:
    notebook_output_path = run_notebook('2.mutation-classifier', timeout=3*60*60)
    print('Machine learning completed.')
except MemoryError as error:
    print(error)
except Exception as error:
    print('Failed to complete classifier.')
    print(error)

print('Check notebook output in folder output.')
