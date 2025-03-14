from keras._tf_keras.keras.models import load_model
import numpy as np
from collections import Counter

class_names = [
    'Apple_Scab_leaf',
    'Apple_leaf_healthy',
    'Apple_rust_leaf',
    'Bell_pepper_Bacterial_spot_leaf',
    'Bell_pepper_leaf_healthy',
    'Blueberry_leaf_healthy',
    'Cherry_Powdery_mildew_leaf',
    'Cherry_leaf_healthy',
    'Corn_leaf_blight',
    'Corn_leaf_healthy',
    'Grape_Esca_leaf',
    'Grape_blight_leaf',
    'Grape_leaf_black_rot',
    'Grape_leaf_healthy',
    'Orange_Haunglongbing_leaf',
    'Peach_Bacterial_spot_leaf',
    'Peach_leaf_healthy',
    'Potato_leaf_early_blight',
    'Potato_leaf_healthy',
    'Potato_leaf_late_blight',
    'Raspberry_leaf_healthy',
    'Soyabean_leaf_healthy',
    'Squash_Powdery_mildew_leaf',
    'Strawberry_leaf_healthy',
    'Strawberry_scorch_leaf',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Target_spot_leaf',
    'Tomato_leaf_bacterial_spot',
    'Tomato_leaf_healthy',
    'Tomato_leaf_late_blight',
    'Tomato_leaf_mosaic_virus',
    'Tomato_leaf_yellow_curl_virus',
    'Tomato_mold_leaf',
    'Tomato_powdery_mildew_leaf',
    'Tomato_two_spotted_spider_mites_leaf'
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
