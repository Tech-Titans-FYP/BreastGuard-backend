from tensorflow.keras.models import load_model

# Function to load the image classifier model
def load_image_classifier(model_path='model_h5/Image_Classifier_model.h5'):
    return load_model(model_path)