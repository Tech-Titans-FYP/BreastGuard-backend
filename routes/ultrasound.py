import numpy as np
from flask import request, jsonify
import base64
import io
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from utils.us_util import (load_preprocess_image_classification,
                   predict_subtype, benign_subtype_mapping, malignant_subtype_mapping, make_gradcam_heatmap, 
                   display_gradcam, get_subtype_description, load_and_prep_image_segmentation, 
                   apply_black_lesion_on_white_background, apply_lesion_on_white_background, classes)
from models.us_model import (load_segmentation_model, 
                    load_malignant_subtype_model, load_benign_subtype_model,
                    load_classification_model, predict_with_model)
from models.image_classifier_model import load_image_classifier
from utils.image_classifier_util import (load_preprocess_image_for_classifier, classify_image_modality, image_modalities)

classification_model = load_classification_model()
benign_subtype_model = load_benign_subtype_model()
malignant_subtype_model = load_malignant_subtype_model()
segmentation_model = load_segmentation_model()
image_classifier_model = load_image_classifier()

def ultrasound_image_modality():
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
    processed_image_modality = load_preprocess_image_for_classifier(image_np)
    modality_index = classify_image_modality(processed_image_modality, image_classifier_model)
    predicted_modality = image_modalities[modality_index]
    print("Predicted modality:", predicted_modality)

    if predicted_modality == 'MRI':
        return jsonify({'message': 'The submitted image could not be confidently classified as an ultrasound image.', 'isUltrasound': False}), 400

    processed_image_classification = load_preprocess_image_classification(image_np)
    prediction = predict_with_model(classification_model, processed_image_classification)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = classes[predicted_class_index]
    print("Predicted class:", predicted_class_name)
    confidence = np.max(prediction)

    CONFIDENCE_THRESHOLD = 0.9
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({'message': 'The submitted image could not be confidently classified as an ultrasound image.', 'isUltrasound': False}), 400

    subtype_full_name = "Not applicable"
    subtype_description = "Not applicable"
    features = {}
    gradcam_image = None
    processed_original_image_base64 = None
    processed_mask_image_base64 = None

    if predicted_class_name in ['Malignant', 'Benign']:
        if predicted_class_name == 'Malignant':
            subtype_full_name = predict_subtype(malignant_subtype_model, image_np, malignant_subtype_mapping)
        else:
            subtype_full_name = predict_subtype(benign_subtype_model, image_np, benign_subtype_mapping)

        subtype_description, features = get_subtype_description(subtype_full_name)
        print("Subtype:", subtype_full_name)

        gradcam_img_array = preprocess_input(processed_image_classification)
        heatmap = make_gradcam_heatmap(gradcam_img_array, classification_model, 'conv5_block3_out')
        print('Generating Grad-CAM heatmap...')
        gradcam_image = display_gradcam(image_np, heatmap)

        print("Proceeding to segmentation...")
        processed_image_segmentation = load_and_prep_image_segmentation(image_np)
        pred_mask = predict_with_model(segmentation_model, processed_image_segmentation)
        binary_mask = np.squeeze((pred_mask > 0.5).astype(np.uint8))
        original_image_np = np.array(image_np)

        if binary_mask.ndim != 2:
            print("The binary mask has an unexpected number of dimensions.")
            return jsonify({'message': 'Error in segmentation process'}), 500

        binary_mask_resized = cv2.resize(binary_mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        processed_original_lesion_image = apply_lesion_on_white_background(original_image_np, binary_mask_resized)
        processed_mask_lesion_image = apply_black_lesion_on_white_background(original_image_np, binary_mask_resized)

        processed_original_image_pil = Image.fromarray(processed_original_lesion_image)
        processed_mask_image_pil = Image.fromarray(processed_mask_lesion_image)

        original_image_buffer = io.BytesIO()    
        processed_original_image_buffer = io.BytesIO()
        processed_mask_image_buffer = io.BytesIO()
        image_data.save(original_image_buffer, format="JPEG")
        processed_original_image_pil.save(processed_original_image_buffer, format="JPEG")
        processed_mask_image_pil.save(processed_mask_image_buffer, format="JPEG")
        
        processed_original_image_base64 = base64.b64encode(processed_original_image_buffer.getvalue()).decode('utf-8')
        processed_mask_image_base64 = base64.b64encode(processed_mask_image_buffer.getvalue()).decode('utf-8')
    else:
        gradcam_image = "Not applicable"
        processed_original_image_base64 = "Not applicable"
        processed_mask_image_base64 = "Not applicable"

    results = {
        'classification': predicted_class_name,
        'subtype': subtype_full_name,
        'subtype_description': subtype_description,
        'features': features,
        'gradcam': gradcam_image if gradcam_image is not None else "Unknown",
        'processed_original_image': processed_original_image_base64 if processed_original_image_base64 is not None else "Unknown",
        'processed_mask_image': processed_mask_image_base64 if processed_mask_image_base64 is not None else "Unknown"
    }
    print(results)
    return jsonify(results)