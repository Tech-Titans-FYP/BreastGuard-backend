 # histo_model.py

import numpy as np
from tensorflow.keras.models import load_model

def load_classification_model(model_path='model_h5/histopathology/best_model_weights.h5'):
    return load_model(model_path)

def load_subtype_model(model_path='model_h5/histopathology/best_model_weights_subtype.h5'):
    return load_model(model_path)