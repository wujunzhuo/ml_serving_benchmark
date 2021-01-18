import time
import grpc
import locust
from environs import Env
from ai_serving_pb2 import RecordSpec, Record, PredictRequest, Value, ListValue
from ai_serving_pb2_grpc import DeploymentServiceStub


env = Env()
HOST = env.str('HOST', 'localhost:9091')


class Test(locust.User):

    def on_start(self):
        channel = grpc.insecure_channel(HOST)
        self.stub = DeploymentServiceStub(channel)

        num_data = [
            None, 2, None, 0.0, None, None, 0.0, 450.0, 1.0, None, 0.0, None,
            1.0
        ]
        num_values = [Value(number_value=x) for x in num_data]

        cat_data = [
            '05db9164', '08d6d899', '77f2f2e5', 'd16679b9', '25c83c98',
            '7e0ccccf', 'af0809a5', '5b392875', '7cc72ec2', '3b08e48b',
            '9e12e146', '9f32b866', '025225f2', '07d13a8f', '41f10449',
            '31ca40b6', '2005abd1', '698d1c68', '', '', 'dfcfc3fa', '',
            'be7c41b4', 'aee52b6f', '', ''
        ]
        cat_values = [Value(string_value=x) for x in cat_data]

        self.request = PredictRequest(X=RecordSpec(
            records=[Record(fields={
                'num_feat': Value(list_value=ListValue(values=num_values)),
                'cat_feat': Value(list_value=ListValue(values=cat_values))
            })]))
        self.request.model_spec.name = 'trans'

    @locust.task
    def predict(self):
        start_time = time.time()
        try:
            self.stub.Predict(self.request)
        except Exception as e:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request_failure.fire(
                request_type='grpc', name='test', response_time=total_time,
                exception=e)
        else:
            total_time = int((time.time() - start_time) * 1000)
            self.environment.events.request_success.fire(
                request_type='grpc', name='test', response_time=total_time,
                response_length=0)
