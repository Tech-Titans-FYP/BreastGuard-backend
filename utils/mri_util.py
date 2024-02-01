# Import libraries
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
from PIL import Image
import base64
import io

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
        binary_cross_entropy_loss = tf.keras.losses.BinaryFocalCrossentropy()(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * binary_cross_entropy_loss)
        return weighted_loss
    return custom_loss

weight_factor = 0.5
custom_loss_fn = weighted_custom_loss(weight_factor)

def load_preprocess_image(image_np):
    SIZE = 224
    # image = Image.open(image_np).convert('RGB')
    # image = image.resize((SIZE, SIZE))
    # processed_image = custom_preprocessing(np.array(image))
    image = Image.fromarray(image_np).resize((SIZE, SIZE))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def custom_preprocessing(image):
    # Apply your custom preprocessing steps here
    return image  # Return the processed image

# def predict(image, model):
#     prediction = model.predict(image)
#     return np.argmax(prediction, axis=1)[0]  # Returns an integer

def predict(model, image):
    # Check if the image has a batch dimension
    if image.ndim == 3:
        # Add a batch dimension
        image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

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

# def overlay_heatmap(heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
#     heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
#     heatmap = 1 - heatmap  # Invert the heatmap colors
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, colormap)
#     superimposed_img = heatmap * alpha + original_image
#     superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
#     return superimposed_img

def overlay_heatmap(image_np, heatmap, alpha=0.4):
    # Ensure that both heatmap and image_np are of type np.ndarray
    heatmap = np.array(heatmap)
    image_np = np.array(image_np)

    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))

    # Invert the heatmap
    heatmap = 1 - heatmap  # Invert the heatmap colors

    # Convert the heatmap to a color map
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the image
    superimposed_img = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Convert the superimposed image to PIL for resizing and display
    superimposed_img_pil = Image.fromarray(superimposed_img)

    # Resize the image for display purposes
    resized_image = superimposed_img_pil.resize((300, 300), Image.LANCZOS)

    # Convert the resized image to a base64-encoded string for easy embedding or display in web applications
    with io.BytesIO() as buffer:
        resized_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

    return base64_image

def get_img_array(uploaded_file, size):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # Normalizing as per your training script
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

img_size = (224, 224)

def resize_for_display(image, max_size=(300, 300)):
    factor = min(max_size[0] / image.size[0], max_size[1] / image.size[1])
    new_size = (int(factor * image.size[0]), int(factor * image.size[1]))
    return image.resize(new_size, Image.ANTIALIAS)

# def segment_tumor(image, unet_model):
#     SIZE = 128  # The size used for the U-Net model
#     # Resize input image to the size expected by the model
#     resized_image = cv2.resize(image, (SIZE, SIZE))
#     resized_image = resized_image / 255.0  # Normalize the image as done during training
#     resized_image = np.expand_dims(resized_image, axis=0)

#     # Predict the mask
#     mask = unet_model.predict(resized_image)
#     mask = (mask > 0.5).astype(np.uint8)[0, :, :, 0]

#     # Resize mask back to the original image size
#     mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

#     # Create an output image that overlays the white mask on the original image
#     segmented_image = image.copy()
#     segmented_image[mask_resized == 1] = 255  # Apply white color where mask is positive

#     return segmented_image

def preprocess_for_subtype(image_np, size=224):
    image = Image.fromarray(image_np).resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def predict_subtype(model, image, subtype_mapping):
    processed_image = preprocess_for_subtype(image)
    prediction = model.predict(processed_image)
    predicted_subtype_index = np.argmax(prediction, axis=1)[0]
    # Ensure that subtype_mapping is a dictionary with integer keys
    predicted_subtype_name = subtype_mapping.get(predicted_subtype_index, 'Unknown')
    return predicted_subtype_name


subtype_mapping = {
    0: 'Invasive Lobular Carcinoma',
    1: 'No Residual',
    2: 'Paget Disease of the Breast',
    3: 'Mucinous Carcinoma',
    4: 'Invasive Ductal Carcinoma',
}

def convert_to_base64(image_pil):
    img_buffer = io.BytesIO()
    image_pil.save(img_buffer, format="JPEG")
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str