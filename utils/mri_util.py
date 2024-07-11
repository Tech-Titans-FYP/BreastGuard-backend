import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image
import base64
import io
import cv2

# Define the classes for classification
classes = ['Benign', 'Malignant']

# Define the dice_coefficient function
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

# Define the custom precision metric function
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val

# Define the custom recall metric function
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

# Define the weighted_custom_loss function
def weighted_custom_loss(weight_factor):
    def custom_loss(y_true, y_pred):
        tumor_size = tf.reduce_sum(tf.cast(tf.equal(y_true, 255), tf.float32))
        weights = 1 + weight_factor * tumor_size
        binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * binary_cross_entropy_loss)
        return weighted_loss
    return custom_loss

# Function to preprocess images
def load_preprocess_image(image, target_size=(128, 128)):  # Default size for classification model
    image = Image.fromarray(image).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict class
def predict(image, model):
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]

# Function to generate Grad-CAM
def grad_cam(model, image, category_index, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if predictions.shape[1] > 1:
            loss = predictions[:, category_index]
        else:
            loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to overlay heatmap
def overlay_heatmap(original_image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    superimposed_img = heatmap * alpha + original_image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Function to convert PIL image to base64 string
def convert_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read())
    return img_str.decode('utf-8')

# Function to segment tumor
def segment_tumor(image, model, SIZE=128, min_radius=10):
    resized_image = cv2.resize(image, (SIZE, SIZE)) / 255.0
    resized_image = np.expand_dims(resized_image, axis=0)

    mask = model.predict(resized_image)
    mask = (mask > 0.5).astype(np.uint8)[0, :, :, 0]
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Highlight the tumor in the original image
    highlighted_image = image.copy()
    non_tumor_area = (mask_resized == 0)

    # Reduce opacity of non-tumor regions by blending with a black image
    alpha = 0.3  # Change this value to adjust the opacity
    black_image = np.zeros_like(highlighted_image)
    highlighted_image = np.where(non_tumor_area[..., None], cv2.addWeighted(highlighted_image, alpha, black_image, 1 - alpha, 0), highlighted_image)

    # Find contours and draw a red circle around the tumor
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = max(min_radius, int(radius * 1.2))  # Ensure the radius is at least min_radius
        cv2.circle(highlighted_image, center, radius, (255, 0, 0), 2)  # Draw red circle

    return highlighted_image, mask_resized

# Function to calculate maximum diameter of tumor
def calculate_max_diameter(mask, pixel_spacing=0.5):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    max_diameter = np.sqrt(w**2 + h**2)
    max_diameter_mm = max_diameter * pixel_spacing
    return max_diameter, max_diameter_mm

# Function to categorize tumor size
def categorize_tumor_size(size_mm):
    if size_mm < 1:
        return 'T1mi'
    elif 1 <= size_mm <= 5:
        return 'T1a'
    elif 5 < size_mm <= 10:
        return 'T1b'
    elif 10 < size_mm <= 20:
        return 'T1c'
    elif 20 < size_mm <= 50:
        return 'T2'
    else:
        return 'T3'


# Add the calculate_tumor_shape_features and categorize_shape functions
def calculate_tumor_shape_features(mask, pixel_spacing=1.0):
    mask = (mask * 255).astype(np.uint8)  # Convert mask back to 0-255 range
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return {"area": 0, "perimeter": 0, "aspect_ratio": 0, "circularity": 0, "MA": 0, "ma": 0}

    contour = contours[0]
    area = cv2.contourArea(contour) * pixel_spacing**2
    perimeter = cv2.arcLength(contour, True) * pixel_spacing
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

    if len(contour) >= 5:  # Minimum points to fit an ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        MA, ma = MA * pixel_spacing, ma * pixel_spacing
    else:
        MA = ma = 0

    return {"area": area, "perimeter": perimeter, "aspect_ratio": aspect_ratio, "circularity": circularity, "MA": MA, "ma": ma}

def categorize_shape(features):
    circularity = features["circularity"]
    aspect_ratio = features["aspect_ratio"]
    if circularity > 0.8 and 0.8 < aspect_ratio < 1.2:
        return "spherical"
    elif circularity > 0.6 and 0.6 < aspect_ratio < 1.4:
        return "discoidal"
    elif circularity < 0.4 and aspect_ratio > 1.5:
        return "segmental"
    else:
        return "irregular"

def predict_subtype(image, model, target_size=(224, 224)):
    image = cv2.resize(image[0], target_size)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    print(f"Subtype model raw prediction: {prediction}")
    return np.argmax(prediction, axis=1)[0]