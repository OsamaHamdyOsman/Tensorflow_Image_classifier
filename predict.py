import argparse, pathlib
import helper_functions as hf
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An App to run image prediction model')

    parser.add_argument('image_file_path', action='store',
                        help='The image file path', type=pathlib.Path)
    parser.add_argument('model_file_path', action='store',
                help='classifier file path of h5 format', type=pathlib.Path)
    parser.add_argument('--top_k', action='store', dest='top_k', default = 5,
                        help="Top k classes, default is 5", type=int)
    parser.add_argument('--category_names', action="store_true", default=False,
                        help='Display the class name relevant to the label.')

    results = parser.parse_args()

    # open image file and convert it to a numpy array
    im = Image.open(results.image_file_path)
    image = np.asarray(im)
    #Preprocessing the image
    image = hf.process_image(image)
    # load the model
    keras_model = tf.keras.models.load_model(results.model_file_path,
                            custom_objects={'KerasLayer':hub.KerasLayer})
    # reading the json file
    with open('label_map.json', 'r') as f:
        class_names = json.load(f)
    # make predictions
    print("\n\nHello There, Let's use the keras model to predict the flower in a sample image\n")
    print("The image file path you have entered is: {}".format(str(results.image_file_path)))
    print('-'*40)

    np.set_printoptions(formatter={'float_kind':'{:.5f}'.format})

    probs, labels, classes = hf.predict(results.image_file_path, keras_model, top_k=results.top_k)

    print('-'*40)
    if results.top_k > 1:
        for pred in  list(zip(labels, classes, probs)):
            print("Label: {}".format(pred[0]))
            if results.category_names:
                print("Class Name: ", pred[1])
            print("Prediction Probability: ", pred[2])
            print('-'*40)
    else:
        print("Label: {}".format(labels))
        if results.category_names:
            print("Class Name: ", classes)
        print("Prediction Probability: ", probs)
