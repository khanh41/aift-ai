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

        keypoint_locs, keypoint_edges, edge_colors, _ = self._get_keypoints_std()
        temp = np.zeros(keypoint_locs.shape)
        temp[[6, 8, 10]] += 1
        keypoint_locs = keypoint_locs * temp

        self.angle_config = self.get_angle(keypoint_locs[[6, 8, 10]])

    def predict(self, image):
        keypoint_locs, keypoint_edges, edge_colors, _ = self.get_keypoints(image)
        # keypoints_predict = DataPreprocessing().affine_transform(keypoint_locs1, keypoint_locs2)

        angle_predict = self.get_angle(keypoint_locs[[6, 8, 10]])
        keypoint_locs[10] = ScoreAngleCalculate().find_new_point(keypoint_locs[8], keypoint_locs[10],
                                                                 self.angle_config - angle_predict, 1)
        keypoints_with_scores = np.expand_dims(np.expand_dims(np.append(keypoint_locs[:, ::-1],
                                                                        np.ones((17, 1)), axis=1), axis=0), axis=0)

        (keypoint_locs, keypoint_edges, edge_colors) = keypoints_and_edges_for_display(keypoints_with_scores,
                                                                                       1, 1)
        temp = np.zeros(keypoint_locs.shape)
        temp[[6, 8, 10]] += 1
        keypoint_locs = keypoint_locs * temp
        keypoint_edges = keypoint_edges[[8, 9]]
        edge_colors = [edge_colors[x] for x in [8, 9]]

        display_image = tf.expand_dims(image, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoint_locs,
                                                  keypoint_edges, edge_colors)

        return pillow_convert_base64(output_overlay)

    def _get_keypoints_std(self):
        image = load_image("app/ml/base_model/images/pushup_1.jpg")
        return self.get_keypoints(image)

    def get_angle(self, keypoints):
        if len(keypoints) == 3:
            return calculateAngle(*keypoints)
        raise Exception("Error")
