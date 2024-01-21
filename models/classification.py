import tensorflow as tf
from tensorflow.keras.models import load_model

def load_classification_model(model_path='ResNet50_model.h5'):
    return load_model(model_path)