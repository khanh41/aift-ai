import pyximport

pyximport.install()

import time
from concurrent import futures

import grpc

from ops.grpc import infer_pb2_grpc, infer_pb2, infer


class CalculatorServicer(infer_pb2_grpc.ExercisePredictServicer):
    def Inference(self, request, context):
        response = infer_pb2.ImgBase64()
        response.img_origin = infer.inference(request.exercise_name, request.img_origin)
        return response


# create a gRPC server
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

infer_pb2_grpc.add_ExercisePredictServicer_to_server(
    CalculatorServicer(), server)

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
