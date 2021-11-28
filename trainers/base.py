from app.ml.base_model.movenet import movenet
from app.ml.figures.draw_keypoints import keypoints_and_edges_for_display
from app.ml.preprocessing import preprocessing_image_movenet


class BaseTrainer(object):
    type_exercise: str

    def get_keypoints_std(self):
        """"""

    def get_keypoints(self, input_image):
        keypoints_with_scores = movenet(preprocessing_image_movenet(input_image))

        (keypoint_locs, keypoint_edges, edge_colors) = keypoints_and_edges_for_display(keypoints_with_scores,
                                                                                       1280, 1280)
        return keypoint_locs, keypoint_edges, edge_colors, input_image

    def calculate_score(self):
        return 100
