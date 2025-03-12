from keras._tf_keras.keras.utils import img_to_array
import numpy as np
from keras._tf_keras.keras.applications.efficientnet_v2 import preprocess_input


def preprocess_image(image, target_size=(224, 224)):
    img_array = img_to_array(image.convert('RGB').resize(target_size))
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def preprocess_images(images):
    return [preprocess_image(img) for img in images]
