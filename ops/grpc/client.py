import grpc

from ops.grpc import infer_pb2, infer_pb2_grpc
from utils import pillow_convert_base64, read_image_from_url
from utils.constants import FIREBASE_IMAGE_URL

channel = grpc.insecure_channel('localhost:50051')

# create a stub (client)
stub = infer_pb2_grpc.ExercisePredictStub(channel)

# create a valid request message
# img = cv2.imread("test_pose.jpg")

img = read_image_from_url(FIREBASE_IMAGE_URL("pushup1"))
img = pillow_convert_base64(img)

number = infer_pb2.ExercisePredictRequest(exercise_name="pushup1", img_origin=img)

# make the call
response = stub.Inference(number)

print(response.img_origin)
