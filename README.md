# Cognoma ml-workers

This repository houses the machine learning worker code. The workers access the [core-service](https://github.com/cognoma/core-service) using an HTTP RESTful API. The machine learning code used by the workers comes from the [machine-learning](https://github.com/cognoma/machine-learning) repository.

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

### Test

Replace '/ml-workers' with the path to your directory

```sh
docker run -v /ml-workers:/code ml-workers /bin/bash -c "python test_ml_task_runner.py"
```

## Deployment

### Prerequisites

This project is deployed within the Alex's Lemonade Stand Foundation AWS account. To be able
to deploy this project you will need to:
1. Be invited to the account.
2. Receive an AWS access key and secret key.

### Logging Into ECR

This project leverages
[AWS Ec2 Container Service (ECS)](https://aws.amazon.com/ecs/details).
ECS provides a private container registry called the
[Ec2 Container Repository (ECR)](https://aws.amazon.com/ecr/).
To be able to push Docker images to this repository you will first need to
get a login with:
```sh
aws ecr get-login --region us-east-1
```
and then run the output of that command. It will look something like:
```sh
docker login -u AWS -p <A_GIANT_HASH> -e none https://589864003899.dkr.ecr.us-east-1.amazonaws.com
```

### Building, Tagging, and Pushing the Container

Run these commands:
```
docker build --tag cognoma-ml-workers .
docker tag cognoma-ml-workers:latest 589864003899.dkr.ecr.us-east-1.amazonaws.com/cognoma-ml-workers:latest
docker push 589864003899.dkr.ecr.us-east-1.amazonaws.com/cognoma-ml-workers:latest
```

### Restarting the ECS Task

Navigate to
[Cognoma's ECS Tasks Page](https://console.aws.amazon.com/ecs/home?region=us-east-1#/clusters/cognoma/tasks)
and select the tasks corresponding to the ml-workers.
The task will have a **Task Definition** like `cognoma-ml-workers:X`
which can be used to identify the correct
task. Once you have selected the correct tasks click the **Stop** button.
This will cause the tasks to be stopped and ECS will restart them with the
new version of the container you have pushed. Therefore you're now done.
