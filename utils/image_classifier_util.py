import numpy as np
from PIL import Image

# Function to preprocess images for the image classifier
def load_preprocess_image_for_classifier(image_np, size=256):
    image = Image.fromarray(image_np).resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Function to classify the image modality
def classify_image_modality(image, model):
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    return predicted_class_index

# Image modalities
image_modalities = ['Ultrasound', 'MRI']

# Function to preprocess images for the image classifier
def load_preprocess_image_for_classifier_mri(image_np, size=128):
    image = Image.fromarray(image_np).resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image