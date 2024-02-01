import numpy as np
from tensorflow.keras.models import load_model
from utils.mri_util import (
    dice_coefficient, precision, recall, custom_loss_fn
)

def load_classification_mri_model(model_path='model_h5/mri/mri_breast_cancer_model_DenseNet201.h5'):
    return load_model(model_path)

def load_segmentation_mri_model(model_path='model_h5/mri/breadm_unet_segmentation_model.h5'):
    return load_model(model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall,
        'custom_loss': custom_loss_fn
    })

def load_malignant_subtype_mri_model(model_path='model_h5/mri/combined_subtype_model.h5'):
    return load_model(model_path)
