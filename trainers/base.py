from base_model.movenet import movenet
from figures.draw_keypoints import keypoints_and_edges_for_display
from preprocessing import preprocessing_image_movenet


class BaseTrainer(object):
    type_exercise: str

    def get_keypoints_std(self):
        """"""

    def get_keypoints(self, input_image):
        keypoints_with_scores = movenet(preprocessing_image_movenet(input_image))

        keypoint_locs = keypoints_and_edges_for_display(keypoints_with_scores, 1280, 1280)
        return keypoint_locs

    def calculate_score(self):
        return 100
