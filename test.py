import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from base_model.movenet import movenet, input_size
from data_loader import load_image
from figures.draw_keypoints import draw_prediction_on_image, keypoints_and_edges_for_display

now = time.time()
image = load_image("base_model/images/pushup_1.jpg")
# Resize and pad the image to keep the aspect ratio and fit the expected size.
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

# Run model inference.
keypoints_with_scores = movenet(input_image)
print(time.time() - now)

# Calculate the angle between the three landmarks.
# angle = calculateAngle((558, 326, 0), (642, 333, 0), (718, 321, 0))
# Display the calculated angle.
# print(f'The calculated angle is {angle}')

# Visualize the predictions with image.
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)
display_image = np.squeeze(display_image.numpy(), axis=0)
height, width, channel = display_image.shape
(keypoint_locs, keypoint_edges, edge_colors) = keypoints_and_edges_for_display(keypoints_with_scores, height, width)

output_overlay = draw_prediction_on_image(display_image, keypoint_locs, keypoint_edges, edge_colors)

plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
_ = plt.axis('off')
plt.show()
