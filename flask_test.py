import json
import locust
from environs import Env


env = Env()


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:8000')

    def on_start(self):
        data = [0.5 for _ in range(13)] + [1 for _ in range(26)]
        self.request = json.dumps({'inputs': [data]})
        self.headers = {'Content-type': 'application/json'}

    @locust.task
    def predict(self):
        self.client.post(
            '/predict', data=self.request, headers=self.headers, verify=False)
