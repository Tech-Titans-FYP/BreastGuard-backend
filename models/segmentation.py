from tensorflow.keras.models import load_model
from utils.custom_losses import dice_coefficient, precision, recall, custom_loss_fn

def load_segmentation_model(model_path='unet_segmentation_model.h5'):
    return load_model(model_path, custom_objects={
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall,
        'custom_loss': custom_loss_fn
    })