import json
import locust
from environs import Env


env = Env()
DATA_SIZE = env.int('DATA_SIZE', 100)


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:8501')

    def on_start(self):
        self.data = {}
        for i in range(1, 14):
            self.data[f'I{i}'] = [[0.5] for _ in range(DATA_SIZE)]

        for i in range(1, 27):
            self.data[f'C{i}'] = [[1] for _ in range(DATA_SIZE)]

    @locust.task
    def predict(self):
        request = json.dumps({'inputs': self.data})
        self.client.post(
            '/v1/models/dcn:predict', data=request, verify=False)
