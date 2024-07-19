import numpy as np
from flask import request, jsonify
import base64
import io
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.densenet import preprocess_input
import cv2
from models.mri_model import load_classification_mri_model, load_segmentation_ultrasound_model, load_subtype_model
from utils.mri_util import load_preprocess_image, predict, classes, grad_cam, overlay_heatmap, segment_tumor, convert_to_base64, calculate_max_diameter, categorize_tumor_size, calculate_tumor_shape_features, categorize_shape, predict_subtype
from models.image_classifier_model import load_image_classifier_mri
from utils.image_classifier_util import classify_image_modality, image_modalities, load_preprocess_image_for_classifier_mri

# Load models
classification_model = load_classification_mri_model()
segmentation_model = load_segmentation_ultrasound_model()
subtype_model = load_subtype_model()
image_classifier_model = load_image_classifier_mri()

def mri_image_modality():
    data = request.get_json()
    print("Received data:", data)
    
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

    image_data = image_data.convert('RGB')
    image_np = np.array(image_data)

    # Classify image modality
    processed_image_modality = load_preprocess_image_for_classifier_mri(image_np)
    modality_index = classify_image_modality(processed_image_modality, image_classifier_model)
    predicted_modality = image_modalities[modality_index]
    print("Predicted modality:", predicted_modality)

    if predicted_modality == 'Ultrasound':
        return jsonify({'message': 'The submitted image could not be confidently classified as an MRI image.', 'isMRI': False}), 400

    # Ensure target size is 128x128 for classification and segmentation models
    processed_image_classification = load_preprocess_image(image_np, target_size=(128, 128))
    print(f"Processed image shape: {processed_image_classification.shape}")
    
    prediction = classification_model.predict(processed_image_classification)
    print(f"Classification prediction: {prediction}")

    # Add confidence threshold check
    confidence = np.max(prediction)
    CONFIDENCE_THRESHOLD = 0.8
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({'message': 'The submitted image could not be confidently classified as an MRI image.', 'isMRI': False}), 400

    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = classes[predicted_class_index]
    print("Predicted class:", predicted_class_name)

    gradcam_img_array = preprocess_input(processed_image_classification)
    layer_name = 'conv5_block32_2_conv'
    heatmap = grad_cam(classification_model, gradcam_img_array, predicted_class_index, layer_name)

    print('Generating Grad-CAM heatmap...')
    gradcam_image_np = overlay_heatmap(image_np, heatmap)

    gradcam_image_pil = Image.fromarray(gradcam_image_np)
    gradcam_image = convert_to_base64(gradcam_image_pil)

    segmented_image, mask_resized = segment_tumor(image_np, segmentation_model)
    segmented_image_pil = Image.fromarray(segmented_image.astype('uint8'), 'RGB')
    segmented_image_base64 = convert_to_base64(segmented_image_pil)

    # Tumor size measurement
    tumor_size_px, tumor_size_mm = calculate_max_diameter(mask_resized, pixel_spacing=0.5)
    tumor_category = categorize_tumor_size(tumor_size_mm)

    # Tumor shape measurement
    shape_features_px = calculate_tumor_shape_features(mask_resized, pixel_spacing=1.0)
    shape_features_mm = calculate_tumor_shape_features(mask_resized, pixel_spacing=0.5)
    shape_category = categorize_shape(shape_features_mm)
    
    # Subtype prediction
    if predicted_class_name == "Malignant":
        processed_image_subtype = load_preprocess_image(image_np, target_size=(224, 224))
        predicted_subtype_index = predict_subtype(processed_image_subtype, subtype_model, target_size=(224, 224))
        predicted_subtype_name = ["Invasive Ductal Carcinoma", "Invasive Lobular Carcinoma", "Mucinous Carcinoma", "No Residual", "Paget Disease of the Breast"][predicted_subtype_index]  # Replace with actual subtype names
        print(f"Predicted subtype index: {predicted_subtype_index}, Subtype name: {predicted_subtype_name}")
    else:
        predicted_subtype_name = "N/A"

    results = {
        'classification': predicted_class_name,
        'gradcam_image': gradcam_image,
        'segmented_image': segmented_image_base64,
        'tumor_size_px': tumor_size_px,
        'tumor_size_mm': tumor_size_mm,
        'tumor_category': tumor_category,
        'shape_features_px': shape_features_px,
        'shape_features_mm': shape_features_mm,
        'shape_category': shape_category,
        'predicted_subtype': predicted_subtype_name  # Add the subtype prediction result
    }
    print(results)
    return jsonify(results)

