import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import json
from PIL import Image
# Import TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


with open('label_map.json', 'r') as f:
    class_names = json.load(f)

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()

    return image


def predict(image_path, model, top_k=5):
    '''
    Returns the top_k predictions of the image by the
    trained mode along with their probabilities and numeric labels.
    Plots the flower image and an horizontal bar chart of the predictions
    and their probabilities

        Parameters:
            image path, trainded model and top n predictions (default is one) as per their probabilities.

        Returns:
            * top n probabilities of the predictions
            * top n predicted class numbers
            * top n predicted class names
    '''
    # open the image
    im = Image.open(image_path)
    # convert the image to a numpy array
    im = np.asarray(im)
    # process the image
    processed_image = process_image(im)
    # add extra dimension
    expanded_image = np.expand_dims(processed_image, 0)
    # predict the image using the model
    ps = model(expanded_image)

    props = tf.math.top_k(ps, k=top_k).values.numpy().squeeze()
    labels = list(map(str, tf.math.top_k(ps, k=top_k).indices.numpy().squeeze()+np.array([1])))
    classes = [class_names[key].title() for key in labels]

    return props, labels, classes
