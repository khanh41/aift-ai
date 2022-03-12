from figures.draw_keypoints import draw_prediction_on_image
from metrics.score import ScoreAngleCalculate
from trainers.base import BaseTrainer
from utils import calculateAngle, pillow_convert_base64, read_image_from_url, get_all_exercise
from utils.constants import FIREBASE_IMAGE_URL, POINTS_SELECTED


class Trainer(BaseTrainer):
    def __init__(self):
        self.config, self.num_step = get_all_exercise()
        self.actual_images = {}

    def predict(self, exercise_name, image, angle_config, return_base64=True):
        display_image = image.copy()
        keypoint_locs = self.get_keypoints(image)
        for points_selected in self.config[exercise_name]:
            display_image = self.draw_prediction(display_image, keypoint_locs, points_selected, (0, 43, 237))

            # keypoints_predict = DataPreprocessing().affine_transform(keypoint_locs1, keypoint_locs2)
            _angle_config = angle_config[POINTS_SELECTED.index(points_selected)]
            angle_predict = self.get_angle(keypoint_locs[points_selected])
            scores.append(ScoreAngleCalculate.score_calculate)
            if abs(_angle_config - angle_predict) > 60:
                raise Exception('Wrong direction')

            keypoint_locs[points_selected[2]] = ScoreAngleCalculate().find_new_point(
                keypoint_locs[points_selected[1]].copy(), keypoint_locs[points_selected[2]].copy(),
                _angle_config - angle_predict, 1)

            display_image = self.draw_prediction(display_image, keypoint_locs, points_selected, (73, 235, 52))

        if return_base64:
            return pillow_convert_base64(display_image[:, :, ::-1]), sum(scores) / len(scores)
        return display_image, sum(scores) / len(scores)

    def read_image_from_url_by_exercise_name(self, exercise_name: str):
        if exercise_name not in self.actual_images.keys():
            self.actual_images[exercise_name] = read_image_from_url(FIREBASE_IMAGE_URL(exercise_name))
        return self.actual_images[exercise_name]

    def get_angle_config(self, actual_image, points_selected):
        keypoint_locs = self.get_keypoints(actual_image)
        if keypoint_locs.shape[0] != 17:
            raise Exception('Not Full Body')

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
        output_image = draw_prediction_on_image(image, keypoint_locs, points_selected, color)

        return output_image


trainer = Trainer()
