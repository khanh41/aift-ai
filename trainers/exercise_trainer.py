import numpy as np
import tensorflow as tf

from figures.draw_keypoints import draw_prediction_on_image
from figures.draw_keypoints import keypoints_and_edges_for_display
from metrics.score import ScoreAngleCalculate
from trainers.base import BaseTrainer
from utils import calculateAngle, pillow_convert_base64, read_image_from_url, get_all_exercise
from utils.constants import FIREBASE_IMAGE_URL


class Trainer(BaseTrainer):
    def __init__(self):
        self.config, self.num_step = get_all_exercise()

    def predict(self, exercise_code, exercise_name, image, return_base64=True):
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        display_image = np.squeeze(display_image.numpy(), axis=0)

        keypoint_locs = self.get_keypoints(image)

        for points_selected in self.config[exercise_name]:
            display_image = self.draw_prediction(display_image, keypoint_locs, points_selected, (237, 43, 0))

            # keypoints_predict = DataPreprocessing().affine_transform(keypoint_locs1, keypoint_locs2)
            actual_image = self.read_image_from_url_by_exercise_name(exercise_code)
            angle_config = self.get_angle_config(actual_image, points_selected)
            angle_predict = self.get_angle(keypoint_locs[points_selected])
            keypoint_locs[points_selected[2]] = ScoreAngleCalculate().find_new_point(
                keypoint_locs[points_selected[1]].copy(), keypoint_locs[points_selected[2]].copy(),
                angle_config - angle_predict, 1)

            display_image = self.draw_prediction(display_image, keypoint_locs, points_selected, (73, 235, 52))

        if return_base64:
            return pillow_convert_base64(display_image[:, :, ::-1])
        return display_image

    @staticmethod
    def read_image_from_url_by_exercise_name(exercise_name: str):
        print(FIREBASE_IMAGE_URL(exercise_name))
        return read_image_from_url(FIREBASE_IMAGE_URL(exercise_name))

    def get_angle_config(self, actual_image, points_selected):
        keypoint_locs = self.get_keypoints(actual_image)
        if len(keypoint_locs) == 0:
            return None

        temp = np.zeros(keypoint_locs.shape)
        temp[points_selected] += 1
        keypoint_locs = keypoint_locs * temp

        angle_config = self.get_angle(keypoint_locs[points_selected])
        return angle_config

    @staticmethod
    def get_angle(keypoints):
        if len(keypoints) == 3:
            return calculateAngle(*keypoints)
        raise Exception("Error")

    @staticmethod
    def draw_prediction(image, keypoint_locs, points_selected: list, color):
        if keypoint_locs.shape[0] != 17:
            raise Exception('Not Full Body')
        keypoints_with_scores = np.expand_dims(np.expand_dims(np.append(keypoint_locs[:, ::-1],
                                                                        np.ones((17, 1)), axis=1), axis=0), axis=0)

        keypoint_locs = keypoints_and_edges_for_display(keypoints_with_scores, 1, 1)
        temp = np.zeros(keypoint_locs.shape)
        temp[points_selected] += 1
        keypoint_locs = keypoint_locs * temp

        output_overlay = draw_prediction_on_image(image, keypoint_locs, points_selected, color[::-1])
        return output_overlay


trainer = Trainer()
