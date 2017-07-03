FROM python:3.6-slim

RUN apt-get update && \
	apt-get install -y wget git nano grep sed && \
    apt-get clean

RUN mkdir /code && mkdir /output
WORKDIR /code
ADD . /code/

RUN git clone https://github.com/cognoma/machine-learning.git cognoma-machine-learning && \
	cd cognoma-machine-learning && git reset --hard 57f7bd016761baf80af812b91d456aef63b8de08 && cd .. && \
	pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1

ENTRYPOINT /bin/bash -c "python ml_task_runner.py"