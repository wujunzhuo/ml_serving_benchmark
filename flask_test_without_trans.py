import json
import locust
from environs import Env


env = Env()


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:8000')

    def on_start(self):
        self.data = [0.5 for _ in range(13)] + [1 for _ in range(26)]
        self.headers = {'Content-Type': 'application/json'}

    @locust.task
    def predict(self):
        request = json.dumps({'inputs': [self.data], 'with_trans': False})
        self.client.post(
            '/predict', data=request, headers=self.headers, verify=False)
