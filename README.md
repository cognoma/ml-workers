# Cognoma ml-workers

This repository houses the machine learning worker code. The workers access the [core-service](https://github.com/cognoma/core-service) and [task-service](https://github.com/cognoma/task-service) using an HTTP RESTful API. The actually machine learning code is housed as an installable module in the [machine-learning](https://github.com/cognoma/machine-learning) repository.

## Getting started

Make sure to fork [this repository on
 +GitHub](https://github.com/cognoma/ml-workers "cognoma/ml-workers on
 +GitHub") first.

### Prerequisites
- Python 3 - tested with Python 3.5.1
- virtualenv - tested on 15.0.2

### Setup

```sh
USERNAME=your_github_handle # Change to your GitHub Handle
git clone git@github.com:${USERNAME}/ml-workers.git
cd ml-workers
virtualenv --python=python3 env
source env/bin/activate
pip install --requirement requirements.txt
```
