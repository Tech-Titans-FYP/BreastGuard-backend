# Import libraries
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
from PIL import Image
import base64
import io

# ---------------------Classification---------------------
classes = ['Benign', 'Malignant', 'Normal']

# Function to load and preprocess the image for classification
def load_preprocess_image_classification(image_np, size=256):
    # Assume image_np is already a NumPy array
    # Resize the image
    image = Image.fromarray(image_np).resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

def resize_image_for_display(image, max_display_size=(300, 300)):
    """
    Resize the uploaded image for display on Streamlit while maintaining the aspect ratio.
    This does not affect the image used for model predictions.
    """
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    if aspect_ratio > 1:
        # Image is wider than it is tall
        new_width = min(image.width, max_display_size[0])
        new_height = int(new_width / aspect_ratio)
    else:
        # Image is taller than it is wide
        new_height = min(image.height, max_display_size[1])
        new_width = int(new_height * aspect_ratio)
        
    return image.resize((new_width, new_height), Image.LANCZOS)

# ---------------------U-Net Segmentation---------------------

# Define dice_coefficient
def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

# Define precision
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_val = true_positives / (predicted_positives + K.epsilon())
    return precision_val

# Define recall
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_val = true_positives / (possible_positives + K.epsilon())
    return recall_val

# Define dice_loss
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# Define weighted_custom_loss
def weighted_custom_loss(weight_factor):
    def custom_loss(y_true, y_pred):
        # Counting the number of white pixels (tumor region) in the true mask
        tumor_size = tf.reduce_sum(tf.cast(tf.equal(y_true, 255), tf.float32))

        # Calculate weights based on tumor size - larger tumors get higher weights
        weights = 1 + weight_factor * tumor_size

        # Standard binary cross-entropy loss
        binary_cross_entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)

        # Apply weights to the loss
        weighted_loss = weights * binary_cross_entropy_loss

        # Return the mean loss
        return tf.reduce_mean(weighted_loss)
    return custom_loss

# Assuming the weight_factor was 0.5 when the model was trained
weight_factor = 0.5
custom_loss_fn = weighted_custom_loss(weight_factor)

def load_and_prep_image_segmentation(image_np, img_shape=128):
    # Assume image_np is already a NumPy array
    img = Image.fromarray(image_np).resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return img

def predict_mask(model, img):
    img_array = np.expand_dims(img, axis=0)  # Add batch dimension
    pred_mask = model.predict(img_array)
    return np.squeeze(pred_mask)  # Remove batch dimension for further processing

def apply_lesion_on_white_background(original_image, binary_mask):
    # Convert binary mask to boolean
    binary_mask_boolean = binary_mask.astype(bool)
    
    # Prepare a white background
    white_background = np.ones_like(original_image) * 255
    
    # Use the binary mask to select the lesion and white background
    result_image = np.where(binary_mask_boolean[..., None], original_image, white_background)
    
    return result_image

def apply_black_lesion_on_white_background(original_image, binary_mask):
    # Invert the binary mask: 1 for lesion, 0 for background
    inverted_mask = 1 - binary_mask
    
    # Create a black lesion area
    black_lesion = np.zeros_like(original_image)
    
    # Combine the black lesion and the inverted mask to get a white background
    result_image = np.where(inverted_mask[..., None], 255, black_lesion)
    
    return result_image

# Define function to resize mask for display
def resize_mask_for_display(mask, display_size=(300, 300)):
    """
    Resize the segmentation mask for display on Streamlit.
    """
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert to an image
    return mask_image.resize(display_size)

def calculate_tumor_size(mask):
    return np.sum(mask)

def calculate_tumor_size_mm2(mask, pixel_density):
    # Convert the size from pixels to square millimeters
    size_in_pixels = calculate_tumor_size(mask)
    size_in_mm2 = size_in_pixels * (pixel_density ** 2)
    return int(size_in_pixels), float(size_in_mm2)  # Ensure these are JSON serializable

def calculate_tumor_volume(masks, slice_thickness=1):
    volumes = []
    for mask in masks:
        size = calculate_tumor_size(mask)
        volume = size * slice_thickness
        volumes.append(volume)
    return float(np.sum(volumes))  # Ensure this is JSON serializable

def get_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    return bounding_boxes

def draw_bounding_boxes_on_image(image, bounding_boxes):
    # Ensure the image is in grayscale before converting
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = image  # Already in BGR format
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red color bounding box with thickness 2
    return image_bgr

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# ---------------------Diagnosis classification---------------------
def preprocess_for_subtype(image_np, size=256):
    image = Image.fromarray(image_np).resize((size, size))
    processed_image = np.array(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    return processed_image

# Dictionary mapping the subtype abbreviations to full names
subtype_full_names = {
    'CYST': 'Cyst',
    'FA': 'Fibroadenoma',
    'LN': 'Lymph Node',
    'PAP': 'Papiloma',
    'DCIS': 'Ductal Carcinoma In Situ',
    'IDC': 'Invasive Ductal Carcinoma',
    'ILC': 'Invasive Lobular Carcinoma',
    'LP': 'Lymphoma',
    'UNK': 'Unknown'
}

benign_subtype_mapping = {
    0: 'Cysts',
    1: 'Fibroadenoma',
    2: 'Galactoceles and sebaceous cysts',
    3: 'Post-operative changes and fat necrosis',
    4: 'Papiloma',
    5: 'Inflammatory conditions'
}

malignant_subtype_mapping = {
    0: 'Carcinoma in situ and microcalcifications',
    1: 'Inflammatory carcinoma',
    2: 'Lymphoma',
    3: 'Ductal Carcinoma In Situ',
    4: 'Invasive Ductal Carcinoma',
    5: 'Invasive Lobular Carcinoma',
    6: 'Metastatic disease'
}

def predict_subtype(model, image, subtype_mapping):
    processed_image = preprocess_for_subtype(image)
    prediction = model.predict(processed_image)
    predicted_subtype_index = np.argmax(prediction, axis=1)[0]
    # Ensure that subtype_mapping is a dictionary with integer keys
    predicted_subtype_name = subtype_mapping.get(predicted_subtype_index, 'Unknown')
    return predicted_subtype_name

subtype_descriptions = {
    'Cysts': {
        'description': 'Benign fluid-filled sacs within the breast, ranging from small cysts mimicking solid lesions to larger thick-walled cysts. They may appear as hypoechoic or anechoic (dark) areas with thin or thick walls and may show signs of infection or vascularity in the wall.',
        'features': {
            'lesion_shape': 'Fluid-filled sacs',
            'margins': 'Thin or thick walls',
            'size': 'Varies from small to large',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Hypoechoic or anechoic areas'
        }
    },
    'Fibroadenoma': {
        'description': 'Benign breast tumors often presenting as multiple, oval-shaped hypoechoic lesions with well-defined edges. They can vary in size and number and may occur in either breast.',
        'features': {
            'lesion_shape': 'Oval-shaped',
            'margins': 'Well-defined edges',
            'size': 'Varies in size and number',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Hypoechoic lesions'
        }
    },
    'Galactoceles and sebaceous cysts': {
        'description': 'Galactoceles are milk-filled cysts that commonly occur during lactation, while sebaceous cysts are small lumps filled with sebum. Both appear as well-circumscribed lesions within the breast tissue.',
        'features': {
            'lesion_shape': 'Well-circumscribed lumps',
            'margins': 'Well-defined',
            'size': 'Small',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Hypoechoic'
        }
    },
    'Post-operative changes and fat necrosis': {
        'description': 'Changes in breast tissue following surgery, which may include features such as oil cysts containing thick material. Fat necrosis may appear as a complex mass or as calcifications within the breast.',
        'features': {
            'lesion_shape': 'Complex mass',
            'margins': 'Not specified',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Contains calcifications'
        }
    },
    'Papilloma': {
        'description': 'Intracanalicular growths that may cause nipple discharge. They are typically small and may not be well visualized on ultrasound.',
        'features': {
            'lesion_shape': 'Intracanalicular growths',
            'margins': 'Not specified',
            'size': 'Small',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Not specified'
        }
    },
    'Inflammatory conditions': {
        'description': 'Conditions such as acute mastitis which can present with edematous (swollen) breast tissue, or hematomas characterized by a mix of hypoechoic and hyperechoic areas indicating fluid and blood collection.',
        'features': {
            'lesion_shape': 'Hematomas',
            'margins': 'Not specified',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Hypoechoic and hyperechoic areas'
        }
    },
    'Carcinoma in situ and microcalcifications': {
        'description': 'Early-stage breast cancer that remains within the ducts or lobules. It is often detected through the presence of microcalcifications on imaging, without forming a palpable lump.',
        'features': {
            'lesion_shape': 'Microcalcifications',
            'margins': 'Not specified',
            'size': 'Not specified',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Not specified'
        }
    },
    'Inflammatory carcinoma': {
        'description': 'A type of breast cancer that can cause the breast to become red, swollen, and inflamed. It often mimics an infection and may involve lymph node metastasis.',
        'features': {
            'lesion_shape': 'Not specified',
            'margins': 'Not specified',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Red, swollen, and inflamed'
        }
    },
    'Lymphoma': {
        'description': 'A rare form of breast cancer that may present as a large, poorly differentiated mass. It can be associated with lymph node metastases, and may not form a discrete mass within the breast.',
        'features': {
            'lesion_shape': 'Poorly differentiated mass',
            'margins': 'Not specified',
            'size': 'Large',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Poorly differentiated'
        }
    },
    'Ductal Carcinoma In Situ': {
        'description': 'A non-invasive condition where abnormal cells are contained in the lining of a breast duct.',
        'features': {
            'lesion_shape': 'Contained within ducts',
            'margins': 'Not invasive',
            'size': 'Not specified',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Abnormal cells'
        }
    },
    'Invasive Ductal Carcinoma': {
        'description': 'The most common type of breast cancer, starting in the milk duct and invading surrounding tissue.',
        'features': {
            'lesion_shape': 'Invasive mass',
            'margins': 'Poorly defined',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Solid mass'
        }
    },
    'Invasive Lobular Carcinoma': {
        'description': 'Cancer that begins in the milk-producing glands and may not form a lump, often occurring in both breasts.',
        'features': {
            'lesion_shape': 'Not always a lump',
            'margins': 'Not specified',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Invasive'
        }
    },
    'Metastatic disease': {
        'description': 'Cancer that has spread from another part of the body to the breast, which may present as a secondary tumor within the breast tissue, sometimes associated with a primary tumor in a different location.',
        'features': {
            'lesion_shape': 'Secondary tumor',
            'margins': 'Not specified',
            'size': 'Varies',
            'depth': 'Not specified',
            'orientation': 'Not specified',
            'texture': 'Varies'
        }
    }
}

def get_subtype_description(subtype_name):
    description_info = subtype_descriptions.get(subtype_name, {})
    description = description_info.get('description', "No description available.")
    features = description_info.get('features', {})
    return description, features

# ---------------------Grad CAM---------------------

def get_img_array(img_path, size):
    # img_path is a file path in this case, but you'll need to modify it to work with an uploaded file if necessary
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
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

def display_gradcam(image_np, heatmap, alpha=0.4):
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

    # Resize the image for display
    resized_image = superimposed_img_pil.resize((300, 300), Image.LANCZOS)

    # Convert the resized image to a base64-encoded string
    with io.BytesIO() as buffer:
        resized_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode()

    return base64_image