FROM python:3.6-slim

RUN apt-get update && \
	apt-get install -y wget git nano grep sed && \
    apt-get clean

# Directories
RUN mkdir /code
WORKDIR /code

# Python environment
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Mount application code
ADD . /code/

ENV PYTHONUNBUFFERED 1

CMD /bin/bash -c "python ml_task_runner.py"