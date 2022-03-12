import os
import time
import uuid

import cv2
import numpy as np

from trainers.exercise_trainer import trainer
from utils import get_exercise_code
from utils.constants import ROOT_PATH, POINTS_SELECTED


class ExerciseDetection:
    def __init__(self):
        self.save_video_dict = {}
        self.__init_std_angle_config__()

    def __init_std_angle_config__(self):
        self.angle_config = {}
        for exercise_name in trainer.config.keys():
            angles_list = []
            for num_step in range(trainer.num_step[exercise_name]):
                exercise_code = get_exercise_code(exercise_name, num_step + 1)
                actual_image = trainer.read_image_from_url_by_exercise_name(exercise_code)
                if actual_image.shape[0] > 1000:
                    width, height = actual_image.shape[0:2]
                    scale = width // 1000
                    actual_image = cv2.resize(actual_image, (height // scale, width // scale))

                angle_list = []
                for point_selected in POINTS_SELECTED:
                    angle_list.append(trainer.get_angle_config(actual_image, point_selected))
                angles_list.append(angle_list)
            self.angle_config[exercise_name] = np.array(angles_list)

    def detect_video(self, exercise_name, video_path):
        print('Detecting...')

        file_path = 'resources/videos/split--character' + str(uuid.uuid4()) + '.mp4'
        response_path = os.path.join(ROOT_PATH, file_path)
        video = cv2.VideoCapture(video_path)
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))

        self.save_video_dict[response_path.replace('split--character', '')] = time.time()
        out = cv2.VideoWriter(response_path.replace('split--character', ''),
                              cv2.VideoWriter_fourcc(*'mp4v'), 25,
                              (frame_width, frame_height))

        count_frame = 20
        while True:
            ret, image = video.read()

            if not ret:
                break

            if count_frame % 20 == 0:
                try:
                    now = time.time()
                    img_clf = self.similarity_angle_config(exercise_name, image[:, :, ::-1])
                    print(time.time() - now)
                    if img_clf != -1:
                        exercise_code = get_exercise_code(exercise_name, img_clf)
                        image, _ = self.detect_image(exercise_code, exercise_name, image, False)
                        for i in range(15):
                            out.write(image.astype(np.uint8))
                except:
                    pass

            count_frame += 1
            out.write(image.astype(np.uint8))

        video.release()
        out.release()
        self.remove_old_video()

        print("Done")
        return response_path

    def detect_image(self, exercise_code, exercise_name, image, return_base64=True):
        angle_config = self.angle_config[exercise_name][int(exercise_code[-1]) - 1]
        return trainer.predict(exercise_name, image, angle_config, return_base64)

    def similarity_angle_config(self, exercise_name, image):
        image_clf = -1
        for index, angle_config in enumerate(self.angle_config[exercise_name]):
            angle_predict = []
            for point_selected in POINTS_SELECTED:
                temp = trainer.get_angle_config(image, point_selected)
                if temp:
                    angle_predict.append(temp)
            angle_predict = np.array(angle_predict)

            if angle_config.shape == angle_predict.shape:
                temp = list(np.where(np.abs(angle_config - angle_predict) < 10, 1, 0))
                print(temp.count(0))
                if temp.count(0) < 4:
                    image_clf = index + 1

        return image_clf

    def get_angle_follow_name_exercise(self, exercise_name):
        pass

    def remove_old_video(self):
        remove_keys = []
        for k, v in self.save_video_dict.items():
            if time.time() - v > 1000:
                os.remove(k)
                remove_keys.append(k)

        for k in remove_keys:
            self.save_video_dict.pop(k)


exercise_detect = ExerciseDetection()

if __name__ == '__main__':
    _new_video = exercise_detect.detect_video('Push Up', '/Users/bap/Desktop/pushup-record.mov')
    print()
