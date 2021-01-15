import time
import grpc
import locust
import tensorflow as tf
from environs import Env
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


env = Env()
MODEL_NAME = env.str('MODEL_NAME', 'dcn')
HOST = env.str('HOST', 'localhost:8500')


class Test(locust.User):

    def on_start(self):
        channel = grpc.insecure_channel(HOST)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = MODEL_NAME

        data = tf.make_tensor_proto([[0.5]], dtype=tf.float32)
        for i in range(1, 14):
            self.request.inputs[f'I{i}'].CopyFrom(data)

        data = tf.make_tensor_proto([[1]], dtype=tf.int32)
        for i in range(1, 27):
            self.request.inputs[f'C{i}'].CopyFrom(data)

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
