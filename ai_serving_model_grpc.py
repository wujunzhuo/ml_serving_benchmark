import time
import grpc
import locust
from environs import Env
from ai_serving_pb2 import RecordSpec, Record, PredictRequest, Value, ListValue
from ai_serving_pb2_grpc import DeploymentServiceStub


env = Env()
MODEL_NAME = env.str('MODEL_NAME', 'lgb')
HOST = env.str('HOST', 'localhost:9091')
DATA_SIZE = env.int('DATA_SIZE', 100)


class Test(locust.User):

    def on_start(self):
        channel = grpc.insecure_channel(HOST)
        self.stub = DeploymentServiceStub(channel)

        data = [0.5 for _ in range(13)] + [1 for _ in range(26)]
        values = ListValue(values=[Value(number_value=x) for x in data])
        values = Value(list_value=values)
        values = ListValue(values=[values for _ in range(DATA_SIZE)])
        self.request = PredictRequest(X=RecordSpec(
            records=[Record(fields={
                'num_feat': Value(list_value=values)
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
