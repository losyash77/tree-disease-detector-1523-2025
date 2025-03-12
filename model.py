from keras._tf_keras.keras.models import load_model
import numpy as np
from collections import Counter

class_names = [
    "Apple Cedar Rust (Leaf)",
    "Apple Leaf (Healthy)",
    "Apple Scab (Leaf)",
    "Bell Pepper Bacterial Spot (Leaf)",
    "Bell Pepper Leaf (Healthy)",
    "Blueberry Leaf (Healthy)",
    "Cherry Leaf (Healthy)",
    "Cherry Powdery Mildew (Leaf)",
    "Corn Leaf Blight",
    "Corn Leaf (Healthy)",
    "Grape Esca (Leaf)",
    "Grape Black Rot (Leaf)",
    "Grape Leaf Blight",
    "Grape Leaf (Healthy)",
    "Orange Huanglongbing (Leaf)",
    "Peach Bacterial Spot (Leaf)",
    "Peach Leaf (Healthy)",
    "Potato Early Blight (Leaf)",
    "Potato Leaf (Healthy)",
    "Potato Late Blight (Leaf)",
    "Raspberry Leaf (Healthy)",
    "Soybean Leaf (Healthy)",
    "Squash Powdery Mildew (Leaf)",
    "Strawberry Leaf (Healthy)",
    "Strawberry Leaf Scorch",
    "Tomato Early Blight (Leaf)",
    "Tomato Bacterial Spot (Leaf)",
    "Tomato Leaf (Healthy)",
    "Tomato Late Blight (Leaf)",
    "Tomato Mosaic Virus (Leaf)",
    "Tomato Mold (Leaf)",
    "Tomato Septoria Spot (Leaf)",
    "Tomato Target Spot (Leaf)",
    "Tomato Two-Spotted Spider Mites (Leaf)",
    "Tomato Yellow Curl Virus (Leaf)"
]

weights_paths = [
    "weights/part_1_model.h5",
    "weights/part_2_model.h5",
    "weights/part_3_model.h5",
    "weights/part_4_model.h5",
    "weights/part_5_model.h5",
    "weights/part_6_model.h5"
]


def load_models():
    models = []
    for path in weights_paths[:6]:
        model = load_model(path)
        models.append(model)
    return models


def predict(images, models):
    global_predictions = []

    for image in images:
        predictions = []

        for model in models:
            prediction = model.predict(image)
            predicted_class_idx = np.argmax(prediction)
            predictions.append(predicted_class_idx)

        vote_count = Counter(predictions)
        final_class_idx = vote_count.most_common(1)[0][0]
        global_predictions.append(final_class_idx)

    global_vote_count = Counter(global_predictions)
    return class_names[global_vote_count.most_common(1)[0][0]]
