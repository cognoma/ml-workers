FROM python:3.5.1
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
ADD . /code/
RUN pip install -r requirements.txt
RUN pip install https://github.com/cognoma/cognoml/archive/master.zip
