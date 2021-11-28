from app.ml.data_loader import load_image
from app.ml.trainers.push_up import PushUpTrainer

trainer = PushUpTrainer()

image = load_image("test_pose.jpg")
trainer.predict(image)
