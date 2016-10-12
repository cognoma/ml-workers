import requests

class BaseAPIClient(object):
    def __init__(self, baseurl, auth_token):
        self.baseurl = baseurl
        self.auth_token = auth_token

    def request(self, method, path, **kwargs):
        if 'headers' not in kwargs:
            kwargs['headers'] = {}

        kwargs['headers']['Authorization'] = 'Bearer ' + self.auth_token

        response = requests.request(method, self.baseurl + path, **kwargs)

        ## TODO: handle non-200 responses
        ## TODO: backoff on 503/504 ?

        return response.json()
