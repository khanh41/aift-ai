from base_model.exercise_detect import exercise_detect
from trainers.exercise_trainer import trainer
import time
from utils import stringToRGB


def image_inference(exercise_code, exercise_name, img_origin):
    img = stringToRGB(img_origin)
    now = time.time()
    response, score = exercise_detect.detect_image(exercise_code, exercise_name, img)
    print(time.time() - now)
    return response, str(score)


def video_inference(exercise_name, user_video_path):
    response = exercise_detect.detect_video(exercise_name, user_video_path)
    return response


def update_config(temp):
    trainer.get_all_exercise()
    return 'success'
