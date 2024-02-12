import numpy as np
from flask import request, jsonify
import base64
import io
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2
from utils.us_util import (load_preprocess_image_classification, predict_subtype, benign_subtype_mapping, 
                   malignant_subtype_mapping, make_gradcam_heatmap, display_gradcam, get_subtype_description,
                   load_and_prep_image_segmentation, apply_black_lesion_on_white_background, apply_lesion_on_white_background,
                   classes)
from models.us_model import (load_segmentation_model, 
                    load_malignant_subtype_model, load_benign_subtype_model,
                    load_classification_model, predict_with_model)

classification_model = load_classification_model()
benign_subtype_model = load_benign_subtype_model()
malignant_subtype_model = load_malignant_subtype_model()
segmentation_model = load_segmentation_model()

def ultrasound_image_modality():
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
    processed_image_classification = load_preprocess_image_classification(image_np)
    prediction = predict_with_model(classification_model, processed_image_classification)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Extract the first element
    predicted_class_name = classes[predicted_class_index]
    print("Predicted class:", predicted_class_name)
    confidence = np.max(prediction)

    # Define a confidence threshold
    CONFIDENCE_THRESHOLD = 0.7  # Adjust based on your model's performance and requirements

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({'message': 'The submitted image could not be confidently classified as an ultrasound image.', 'isUltrasound': False}), 400

    # Subtype identification
    if predicted_class_name in ['Malignant', 'Benign']:
        # ---------------------Diagnosis classification---------------------
        if predicted_class_name == 'Malignant':
            subtype_full_name = predict_subtype(malignant_subtype_model, image_np, malignant_subtype_mapping)
        else:
            subtype_full_name = predict_subtype(benign_subtype_model, image_np, benign_subtype_mapping)

        subtype_description = get_subtype_description(subtype_full_name)

        print("Subtype:", subtype_full_name)

        # ---------------------Applying Grad CAM---------------------
        # Preprocess the image for Grad-CAM
        # gradcam_img_array = preprocess_input(get_img_array(uploaded_file, img_size))
        gradcam_img_array = preprocess_input(processed_image_classification)

        # Generate heatmap
        heatmap = make_gradcam_heatmap(gradcam_img_array, classification_model, 'conv5_block3_out')

        # Display Grad-CAM heatmap
        print('Generating Grad-CAM heatmap...')
        gradcam_image = display_gradcam(image_np, heatmap)

        # ---------------------U-Net Segmentation---------------------    
        print("Proceeding to segmentation...")
        processed_image_segmentation = load_and_prep_image_segmentation(image_np)
        pred_mask = predict_with_model(segmentation_model, processed_image_segmentation)

        # Remove the batch and channel dimensions from the binary mask
        binary_mask = np.squeeze((pred_mask > 0.5).astype(np.uint8))

        original_image_np = np.array(image_np)

        # Check if the dimensions are correct before resizing
        if binary_mask.ndim != 2:
            print("The binary mask has an unexpected number of dimensions.")
            return jsonify({'message': 'Error in segmentation process'}), 500

        # Now that we've confirmed the binary_mask is 2D, we can resize it
        binary_mask_resized = cv2.resize(binary_mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Process the binary mask and prepare the images for each task
        processed_original_lesion_image = apply_lesion_on_white_background(original_image_np, binary_mask_resized)
        processed_mask_lesion_image = apply_black_lesion_on_white_background(original_image_np, binary_mask_resized)

        # Convert the processed image back to PIL Image for display in Flask
        processed_original_image_pil = Image.fromarray(processed_original_lesion_image)
        processed_mask_image_pil = Image.fromarray(processed_mask_lesion_image)


        # Encode the images to base64 to send in the JSON response
        original_image_buffer = io.BytesIO()    
        processed_original_image_buffer = io.BytesIO()
        processed_mask_image_buffer = io.BytesIO()
        image_data.save(original_image_buffer, format="JPEG")
        processed_original_image_pil.save(processed_original_image_buffer, format="JPEG")
        processed_mask_image_pil.save(processed_mask_image_buffer, format="JPEG")
        
        processed_original_image_base64 = base64.b64encode(processed_original_image_buffer.getvalue()).decode('utf-8')
        processed_mask_image_base64 = base64.b64encode(processed_mask_image_buffer.getvalue()).decode('utf-8')
    else:
        # Handle the 'Normal' or other cases
        subtype_full_name = "Not applicable"  # or another appropriate default value
        subtype_description = "Not applicable"
        gradcam_image = "Not applicable"
        processed_original_image_base64 = "Not applicable"
        processed_mask_image_base64 = "Not applicable"

    # Combine results and send back
    results = {
        'classification': predicted_class_name,
        'subtype': subtype_full_name if subtype_full_name is not None else "Unknown",
        'subtype_description': subtype_description if subtype_description is not None else "Unknown",
        'gradcam': gradcam_image if gradcam_image is not None else "Unknown",
        'processed_original_image': processed_original_image_base64 if processed_original_image_base64 is not None else "Unknown",
        'processed_mask_image': processed_mask_image_base64 if processed_mask_image_base64 is not None else "Unknown"
    }
    print(results)
    return jsonify(results)