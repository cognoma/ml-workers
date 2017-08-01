import requests
import backoff
from settings import fatal_code

class BaseAPIClient(object):
    def __init__(self, base_url, auth_token, worker_id):
        self.base_url = base_url
        self.auth_token = auth_token
        self.worker_id = worker_id

    @backoff.on_exception(backoff.expo,
                          (requests.RequestException,
                           requests.Timeout,
                           requests.ConnectionError),
                          max_tries=5,
                          giveup=fatal_code,
                          factor=2)
    def request(self, method, path, **kwargs):
        if 'headers' not in kwargs:
            kwargs['headers'] = {}

        kwargs['headers']['Authorization'] = 'JWT ' + self.auth_token

        response = requests.request(method, self.base_url + path, **kwargs)

        if response.status_code < 200 or response.status_code > 299:
            print('Response status code: ' + str(response.status_code))
            print('Response content: ' + str(response.content))
            raise Exception('Failed to hit internal service for: ' + method + ' ' + path)

        print(response)

        return response.json()
