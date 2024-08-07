import numpy as np
from tensorflow.keras.models import load_model
from utils.mri_util import dice_coefficient, precision, recall, weighted_custom_loss, classes

def load_classification_mri_model(model_path='model_h5/mri/mri_breast_cancer_model_DenseNet201.h5'):
    return load_model(model_path)

def load_segmentation_ultrasound_model(model_path='model_h5/mri/breadm_unet_segmentation_model.h5'):
    weight_factor = 0.5
    return load_model(model_path, custom_objects={
        'custom_loss': weighted_custom_loss(weight_factor),
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall
    })

def load_subtype_model(model_path='model_h5/mri/histological_subtype_model.h5'):
    return load_model(model_path)
