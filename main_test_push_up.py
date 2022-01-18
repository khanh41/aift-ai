from data_loader import load_image
from trainers.push_up import trainer

image = load_image("test_pose.jpg")
trainer.predict(image)
