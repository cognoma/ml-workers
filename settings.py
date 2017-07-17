import os
import requests
import backoff

DEBUG = os.getenv('WORKER_DEBUG', 'True') == 'True'

def fatal_code(e):
    if e.response and e.response.status_code:
        return 400 <= e.response.status_code < 500

@backoff.on_exception(backoff.expo,
                      (requests.RequestException,
                       requests.Timeout,
                       requests.ConnectionError),
                      max_tries=5,
                      giveup=fatal_code,
                      factor=2)
def get_worker_id() -> str:
    # Returns local or the AWS instance id where this is running
    if DEBUG:
        return 'local'
    else:
        # http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-metadata.html
        return requests.get('http://169.254.169.254/latest/meta-data/instance-id/').text

if DEBUG:
    auth_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzZXJ2aWNlIjoiY29yZSJ9.HHlbWMjo-Y__DGV0DAiCY7u85FuNtY8wpovcZ9ga-oCsLdM2H5iVSz1vKiWK8zxl7dSYltbnyTNMxXO2cDS81hr4ohycr7YYg5CaE5sA5id73ab5T145XEdF5X_HXoeczctGq7X3x9QYSn7O1fWJbPWcIrOCs6T2DrySsYgjgdAAnWnKedy_dYWJ0YtHY1bXH3Y7T126QqVlQ9ylHk6hmFMCtxMPbuAX4YBJsxwjWpMDpe13xbaU0Uqo5N47a2_vi0XzQ_tzH5esLeFDl236VqhHRTIRTKhPTtRbQmXXy1k-70AU1FJewVrQddxbzMXJLFclStIdG_vW1dWdqhh-hQ'
    base_url = 'http://localhost:8000'
    worker_id = '1'
else:
    auth_token = os.getenv('CORE_SERVICE_AUTH_TOKEN')
    base_url = os.getenv('CORE_SERVICE_BASE_URL')
    worker_id = get_worker_id()
