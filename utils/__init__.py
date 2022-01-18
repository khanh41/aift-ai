import base64
import io
import math

import numpy as np
from PIL import Image
from cv2 import cv2


def calculateAngle(landmark1, landmark2, landmark3):
    """
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    """
    # Get the required landmarks coordinates.
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle <= 0:
        # Add 360 to the found angle.
        angle += 360

    # Return the calculated angle.
    return angle


def stringToRGB(base64_string):
    img = base64.b64decode(str(base64_string))
    return read_image_byte(img)


def read_image_byte(image_byte):
    image = Image.open(io.BytesIO(image_byte))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.asarray(image)
    return image


def base64_img(image, convert_color=False):
    image = np.asarray(image)
    if convert_color:
        image = image[:, :, ::-1]
    ret, buffer = cv2.imencode('.png', image)
    image = base64.b64encode(buffer)
    return image


def pillow_convert_base64(image):
    image = Image.fromarray(image.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(img_str)[2:-1]
    return img_str
