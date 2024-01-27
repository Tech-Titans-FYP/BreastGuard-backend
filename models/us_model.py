import numpy as np
from tensorflow.keras.models import load_model
from utils.us_util import dice_coefficient, precision, recall, custom_loss_fn

# ---------------------Classification---------------------

def load_classification_model(model_path='model_h5/ultrasound/ResNet50_model.h5'):
    return load_model(model_path)

# ---------------------U-Net Segmentation---------------------

def load_segmentation_model(model_path='model_h5/ultrasound/unet_segmentation_model.h5'):
    return load_model(model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall,
        'custom_loss': custom_loss_fn
    })

# ---------------------Diagnosis classification---------------------

def load_benign_subtype_model(model_path='model_h5/ultrasound/benign_subtype_model.h5'):
    return load_model(model_path)

def load_malignant_subtype_model(model_path='model_h5/ultrasound/malignant_subtype_model.h5'):
    return load_model(model_path)

# ---------------------Prediction---------------------
def predict_with_model(model, image):
    # Check if the image has a batch dimension
    if image.ndim == 3:
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction