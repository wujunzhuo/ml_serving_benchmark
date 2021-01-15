import json
import locust
from environs import Env


env = Env()
MODEL_NAME = env.str('MODEL_NAME', 'lgb')


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:9090')

    def on_start(self):
        self.data = [0.5 for _ in range(13)] + [1 for _ in range(26)]
        self.headers = {'Content-type': 'application/json'}

    @locust.task
    def predict(self):
        request = json.dumps({'X': [{'float_input': [self.data]}]})
        self.client.post(
            f'/v1/models/{MODEL_NAME}', data=request,
            headers=self.headers, verify=False)
