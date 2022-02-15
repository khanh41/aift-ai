from base_model.movenet import movenet
from figures.draw_keypoints import keypoints_and_edges_for_display
from utils.crop_video import init_crop_region, run_inference


class BaseTrainer(object):
    type_exercise: str

    def get_keypoints_std(self):
        """"""

    def get_keypoints(self, image):
        input_size = 192

        # Load the input image.
        image_height, image_width, _ = image.shape
        crop_region = init_crop_region(image_height, image_width)

        keypoints_with_scores = run_inference(
            movenet, image, crop_region,
            crop_size=[input_size, input_size])

        keypoint_locs = keypoints_and_edges_for_display(keypoints_with_scores,
                                                        image_height, image_width)
        return keypoint_locs

    def calculate_score(self):
        return 100
