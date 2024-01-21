import numpy as np
from tensorflow.keras.models import load_model

def load_benign_subtype_model(model_path='benign_subtype_model.h5'):
    return load_model(model_path)

def load_malignant_subtype_model(model_path='malignant_subtype_model.h5'):
    return load_model(model_path)

def predict_with_model(model, image):
    # Check if the image has a batch dimension
    if image.ndim == 3:
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction