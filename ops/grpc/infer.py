from trainers.push_up import trainer
from utils import stringToRGB


def inference(img_origin):
    img = stringToRGB(img_origin)
    response = trainer.predict(img)
    return response
