import requests
import backoff

def fatal_code(e):
    if e.response and e.response.status_code:
        return 400 <= e.response.status_code < 500

class BaseAPIClient(object):
    def __init__(self, baseurl, auth_token):
        self.baseurl = baseurl
        self.auth_token = auth_token

    @backoff.on_exception(backoff.expo,
                          (requests.exceptions.RequestException,
                           requests.exceptions.Timeout,
                           requests.exceptions.ConnectionError),
                          max_tries=5,
                          giveup=fatal_code,
                          factor=2)
    def request(self, method, path, **kwargs):
        if 'headers' not in kwargs:
            kwargs['headers'] = {}

        kwargs['headers']['Authorization'] = 'JWT ' + self.auth_token

        response = requests.request(method, self.baseurl + path, **kwargs)

        ## TODO: handle non-200 responses

        print(response)

        return response.json()
