from trainers.push_up import trainer
from utils import stringToRGB


def inference(exercise_name, img_origin):
    img = stringToRGB(img_origin)
    response = trainer.predict(exercise_name, img)
    return response
