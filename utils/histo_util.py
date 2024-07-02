import numpy as np
import cv2
from skimage import exposure
from PIL import Image, UnidentifiedImageError
import io
import base64
import tensorflow as tf

def custom_preprocessing(image):
    """
    This function performs custom preprocessing on the input image, including
    whitespace removal, image tiling, normalization, quick balancing, and augmentation.
    """
    target_size = (224, 224)
    image = custom_whitespace_removal(image)
    image = custom_image_tiling(image, target_size)
    image = custom_image_normalization(image)
    image = custom_quick_balancing(image)
    image = custom_augmentation(image)
    return image

def custom_whitespace_removal(image):
    """
    Removes whitespace from the image using a binary threshold.
    """
    threshold = 200
    binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    return cv2.bitwise_and(image, binary_image)

def custom_image_tiling(image, target_size):
    """
    Tiles the image to the target size by cropping and resizing.
    """
    h, w = image.shape[:2]
    th, tw = target_size
    start_h = max(0, (h - th) // 2)
    start_w = max(0, (w - tw) // 2)
    cropped_image = image[start_h:start_h+th, start_w:start_w+tw]
    return cv2.resize(cropped_image, target_size)

def custom_image_normalization(image):
    """
    Normalizes the image pixel values to the range [0, 1].
    """
    if np.min(image) == np.max(image):
        return image
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def custom_quick_balancing(image):
    """
    Applies histogram equalization to balance the image.
    """
    return exposure.equalize_hist(image)

def custom_augmentation(image):
    """
    Applies augmentation by flipping the image horizontally.
    """
    return cv2.flip(image, 1)

def preprocess_for_prediction(image):
    """
    Preprocesses the image for prediction by converting it to RGB, applying custom
    preprocessing, and expanding the dimensions.
    """
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image_np = np.array(image)
    processed_image = custom_preprocessing(image_np)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given model and image.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def plot_gradcam(image, heatmap, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the image and returns it as a base64 string.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlay_image = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    
    overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(overlay_image_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str
