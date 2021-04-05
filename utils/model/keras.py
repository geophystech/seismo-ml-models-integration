from keras.models import model_from_json
import tensorflow as tf

# TODO: remove following code and move GPU support somehow to config settings
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def json_load(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects = {"tf": tf})

    model.load_weights(weights_path)

    return model
