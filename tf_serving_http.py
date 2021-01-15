import json
import locust
from environs import Env


env = Env()
MODEL_NAME = env.str('MODEL_NAME', 'dcn')


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:8501')

    def on_start(self):
        self.data = {}
        for i in range(1, 14):
            self.data[f'I{i}'] = [[0.5]]

        for i in range(1, 27):
            self.data[f'C{i}'] = [[1]]

    @locust.task
    def predict(self):
        request = json.dumps({'inputs': self.data})
        self.client.post(
            f'/v1/models/{MODEL_NAME}:predict', data=request,
            verify=False)
