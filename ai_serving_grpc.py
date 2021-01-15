import time
import grpc
import locust
from environs import Env
from ai_serving_pb2 import RecordSpec, Record, PredictRequest, Value, ListValue
from ai_serving_pb2_grpc import DeploymentServiceStub


env = Env()
MODEL_NAME = env.str('MODEL_NAME', 'lgb')
HOST = env.str('HOST', 'localhost:9091')


class Test(locust.User):

    def on_start(self):
        channel = grpc.insecure_channel(HOST)
        self.stub = DeploymentServiceStub(channel)

        values = [Value(number_value=0.5) for _ in range(13)] + \
            [Value(number_value=1) for _ in range(26)]
        self.request = PredictRequest(X=RecordSpec(
            records=[Record(fields={
                'float_input': Value(list_value=ListValue(values=values))
            })]))
        self.request.model_spec.name = MODEL_NAME

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
