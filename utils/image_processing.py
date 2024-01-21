from PIL import Image
import numpy as np

def load_preprocess_image_classification(uploaded_file, size=256):
    # Read and resize the image
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def load_and_prep_image_segmentation(uploaded_file, img_shape=128):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return img