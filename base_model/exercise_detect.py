import os
import time
import uuid

import cv2
import numpy as np

from trainers.exercise_trainer import trainer
from utils import get_exercise_code
from utils.constants import ROOT_PATH


class ExerciseDetection:
    def __init__(self):
        self.points_selected = [[4, 6, 8], [6, 8, 10], [8, 6, 12], [6, 12, 14], [12, 14, 16],
                                [4, 5, 7], [5, 7, 9], [7, 5, 11], [5, 11, 13], [11, 13, 15]]
        self.__init_std_angle_config__()

    def __init_std_angle_config__(self):
        self.angle_config = {}
        for exercise_name in trainer.config.keys():
            angles_list = []
            for num_step in range(trainer.num_step[exercise_name]):
                exercise_code = get_exercise_code(exercise_name, num_step + 1)
                actual_image = trainer.read_image_from_url_by_exercise_name(exercise_code)
                angle_list = []
                for point_selected in self.points_selected:
                    angle_list.append(trainer.get_angle_config(actual_image, point_selected))
                angles_list.append(angle_list)
            self.angle_config[exercise_name] = np.array(angles_list)

    def detect_video(self, exercise_name, video_path):
        print('Detecting...')

        file_path = 'app/resources/videos/split--character' + str(uuid.uuid4()) + '.avi'
        response_path = os.path.join(ROOT_PATH, file_path)

        out = cv2.VideoWriter(response_path.replace('split--character', ''),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (1280, 1280))

        video = cv2.VideoCapture(video_path)
        count_frame = 5
        while True:
            ret, image = video.read()
            if not ret:
                break

            if count_frame % 5 == 0:
                now = time.time()
                img_clf = self.similarity_angle_config(exercise_name, image)
                print(time.time() - now)
                if img_clf != -1:
                    exercise_code = get_exercise_code(exercise_name, img_clf)
                    image = self.detect_image(exercise_code, exercise_name, image, False)
            count_frame += 1
            out.write(image)

        video.release()
        out.release()

        return response_path

    @staticmethod
    def detect_image(exercise_code, exercise_name, image, return_base64=True):
        return trainer.predict(exercise_code, exercise_name, image, return_base64)

    def similarity_angle_config(self, exercise_name, image):
        image_clf = -1
        for index, angle_config in enumerate(self.angle_config[exercise_name]):
            angle_predict = []
            for point_selected in self.points_selected:
                temp = trainer.get_angle_config(image, point_selected)
                if temp:
                    angle_predict.append(temp)
            angle_predict = np.array(angle_predict)

            if angle_config.shape == angle_predict.shape:
                temp = list(np.where(np.abs(angle_config - angle_predict) < 10, 1, 0))
                print(temp.count(0))
                if temp.count(0) < 5:
                    image_clf = index + 1

        return image_clf

    def get_angle_follow_name_exercise(self, exercise_name):
        pass


exercise_detect = ExerciseDetection()

if __name__ == '__main__':
    _new_video = exercise_detect.detect_video('Push Up', '/Users/bap/Desktop/pushup-record.mov')
    print()
