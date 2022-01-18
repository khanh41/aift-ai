import cv2
import grpc

from ops.grpc import infer_pb2, infer_pb2_grpc
from utils import pillow_convert_base64

channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = infer_pb2_grpc.ExercisePredictStub(channel)

# create a valid request message
img = cv2.imread("test_pose.jpg")
img = pillow_convert_base64(img)
number = infer_pb2.ImgBase64(img_origin=img)

# make the call
response = stub.Inference(number)

print(response.value)
