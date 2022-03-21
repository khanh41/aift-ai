import pyximport

pyximport.install()

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import time
from concurrent import futures

import grpc

from ops.grpc import infer_pb2_grpc, infer_pb2, infer


class ImagePredictServicer(infer_pb2_grpc.ExerciseImagePredictServicer):
    def ImageInference(self, request, context):
        response = infer_pb2.Param2Request()
        response.param_1, response.param_2 = infer.image_inference(request.param_1,
                                                                   request.param_2,
                                                                   request.param_3)
        return response


class VideoPredictServicer(infer_pb2_grpc.ExerciseVideoPredictServicer):
    def VideoInference(self, request, context):
        response = infer_pb2.Param1Request()
        response.param_1 = infer.video_inference(request.param_1,
                                                 request.param_2)
        return response


class UpdateConfigServicer(infer_pb2_grpc.UpdateDataConfigServicer):
    def UpdateConfig(self, request, context):
        response = infer_pb2.Param1Request()
        response.param_1 = infer.update_config('')
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

infer_pb2_grpc.add_ExerciseImagePredictServicer_to_server(ImagePredictServicer(), server)
infer_pb2_grpc.add_ExerciseVideoPredictServicer_to_server(VideoPredictServicer(), server)
infer_pb2_grpc.add_UpdateDataConfigServicer_to_server(UpdateConfigServicer(), server)

# listen on port 50051
print()
print('Starting server. Listening on port 50051.')
server.add_insecure_port('[::]:50051')
server.start()

# since server.start() will not block,
# a sleep-loop is added to keep alive
try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)
