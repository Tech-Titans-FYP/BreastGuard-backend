import numpy as np
from flask import request, jsonify
import base64
import io
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from models.mri_model import (
    load_classification_mri_model, load_malignant_subtype_mri_model, load_segmentation_mri_model
)
from utils.mri_util import (
    load_preprocess_image, predict, classes, grad_cam, overlay_heatmap, predict_subtype, 
    subtype_mapping, segment_tumor, convert_to_base64
)

classification_model = load_classification_mri_model()
subtype_model = load_malignant_subtype_mri_model()
segmentation_model = load_segmentation_mri_model()

def mri_image_modality():
    data = request.get_json()
    print("Received data:", data)
    
    # Check if image data is present and in the correct format
    if 'image' not in data or not isinstance(data['image'], list) or len(data['image']) == 0:
        return jsonify({'message': 'No image data found in the request'}), 400
    
    image_info = data['image'][0]
    if 'url' not in image_info:
        return jsonify({'message': 'Image URL is missing'}), 400

    base64_image_data = image_info['url'].split(';base64,')[-1]
    if not base64_image_data:
        return jsonify({'message': 'Base64 image data is missing'}), 400

    try:
        image_data = base64.b64decode(base64_image_data)
        image_data = io.BytesIO(image_data)
        image_data = Image.open(image_data)
    except (base64.binascii.Error, UnidentifiedImageError):
        return jsonify({'message': 'Invalid image data'}), 400

    # Convert the PIL Image to a NumPy array
    image_data = image_data.convert('RGB')
    image_np = np.array(image_data)

    gradcam_image = None
    
    # Classification
    processed_image_classification = load_preprocess_image(image_np)
    prediction = predict(classification_model, processed_image_classification)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Extract the first element
    predicted_class_name = classes[predicted_class_index]
    print("Predicted class:", predicted_class_name)

    # Subtype identification
    if predicted_class_name in ['Malignant', 'Benign']:
        # ---------------------Diagnosis classification---------------------
        if predicted_class_name == 'Malignant':
            subtype_full_name = predict_subtype(subtype_model, image_np, subtype_mapping)

            print("Subtype:", subtype_full_name)

        # ---------------------Applying Grad CAM---------------------
        # Preprocess the image for Grad-CAM
        # gradcam_img_array = preprocess_input(get_img_array(uploaded_file, img_size))
        gradcam_img_array = preprocess_input(processed_image_classification)

        layer_name = 'conv5_block32_2_conv'

        # Generate heatmap
        # heatmap = grad_cam(gradcam_img_array, classification_model, 'conv5_block32_2_conv')
        heatmap = grad_cam(classification_model, gradcam_img_array, predicted_class_index, layer_name)

        # Display Grad-CAM heatmap
        print('Generating Grad-CAM heatmap...')
        gradcam_image = overlay_heatmap(image_np, heatmap)

    # ---------------------U-Net Segmentation---------------------    

    # Combine results and send back
    results = {
        'classification': predicted_class_name,
        # 'subtype': subtype_full_name,
        # 'subtype_description': subtype_description,
        'gradcam': gradcam_image,  # Ensure this is always set, even if to "Not applicable"
        # 'processed_original_image': processed_original_image_base64,
        # 'processed_mask_image': processed_mask_image_base64
    }
    print(results)
    return jsonify(results)