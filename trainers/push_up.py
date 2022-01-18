import numpy as np
import tensorflow as tf

from data_loader import load_image
from figures.draw_keypoints import draw_prediction_on_image
from figures.draw_keypoints import keypoints_and_edges_for_display
from metrics.score import ScoreAngleCalculate
from trainers.base import BaseTrainer
from utils import calculateAngle, pillow_convert_base64


class PushUpTrainer(BaseTrainer):
    def __init__(self):
        self.type_exercise = 'push-up'

        keypoint_locs = self._get_keypoints_std()
        temp = np.zeros(keypoint_locs.shape)
        temp[[6, 8, 10]] += 1
        keypoint_locs = keypoint_locs * temp

        self.angle_config = self.get_angle(keypoint_locs[[6, 8, 10]])

    def predict(self, image):
        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        display_image = np.squeeze(display_image.numpy(), axis=0)

        keypoint_locs = self.get_keypoints(image)

        output_overlay = self._draw_prediction(display_image, keypoint_locs, (237, 43, 0))

        # keypoints_predict = DataPreprocessing().affine_transform(keypoint_locs1, keypoint_locs2)

        angle_predict = self.get_angle(keypoint_locs[[6, 8, 10]])
        keypoint_locs[10] = ScoreAngleCalculate().find_new_point(keypoint_locs[8].copy(), keypoint_locs[10].copy(),
                                                                 self.angle_config - angle_predict, 1)

        output_overlay = self._draw_prediction(output_overlay, keypoint_locs, (73, 235, 52))

        return pillow_convert_base64(output_overlay)

    def _get_keypoints_std(self):
        image = load_image("base_model/images/pushup_1.jpg")
        return self.get_keypoints(image)

    def get_angle(self, keypoints):
        if len(keypoints) == 3:
            return calculateAngle(*keypoints)
        raise Exception("Error")

    def _draw_prediction(self, image, keypoint_locs, color):
        keypoints_with_scores = np.expand_dims(np.expand_dims(np.append(keypoint_locs[:, ::-1],
                                                                        np.ones((17, 1)), axis=1), axis=0), axis=0)

        keypoint_locs = keypoints_and_edges_for_display(keypoints_with_scores, 1, 1)
        temp = np.zeros(keypoint_locs.shape)
        temp[[6, 8, 10]] += 1
        keypoint_locs = keypoint_locs * temp

        output_overlay = draw_prediction_on_image(image, keypoint_locs, [6, 8, 10], color)
        return output_overlay


trainer = PushUpTrainer()