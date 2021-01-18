import json
import locust
from environs import Env


env = Env()


class Test(locust.HttpUser):

    host = env.str('HOST', 'http://localhost:8000')

    def on_start(self):
        self.data = [
            -9.9, 2, -9.9, 0.0, -9.9, -9.9, 0.0, 450.0, 1.0, -9.9, 0.0, -9.9,
            1.0, '05db9164', '08d6d899', '77f2f2e5', 'd16679b9', '25c83c98',
            '7e0ccccf', 'af0809a5', '5b392875', '7cc72ec2', '3b08e48b',
            '9e12e146', '9f32b866', '025225f2', '07d13a8f', '41f10449',
            '31ca40b6', '2005abd1', '698d1c68', '', '', 'dfcfc3fa', '',
            'be7c41b4', 'aee52b6f', '', ''
        ]
        self.headers = {'Content-Type': 'application/json'}

    @locust.task
    def predict(self):
        request = json.dumps({'inputs': [self.data]})
        self.client.post(
            '/predict', data=request, headers=self.headers, verify=False)
