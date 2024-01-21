from flask import Flask, request, jsonify
import numpy as np
from flask import request, jsonify
from models.classification import load_classification_model
from utils.image_processing import load_preprocess_image_classification, load_and_prep_image_segmentation
from utils.gradcam import make_gradcam_heatmap
from models.segmentation import load_segmentation_model
from models.subtype import load_benign_subtype_model
from models.subtype import load_malignant_subtype_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import base64
import io
from PIL import Image, UnidentifiedImageError
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allow CORS for all routes
app.config['UPLOAD_FOLDER'] = 'E:\\L4S1\\IS4910 - Comprehensive Group Project\\FlaskDemo\\uploads'

classes = ['Benign', 'Malignant', 'Normal']

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

@app.route('/api/process-image', methods=['POST'])
def process_image():
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
    
    # Preprocess the image for each model
    # Note: Modify your preprocessing functions to handle NumPy arrays directly
    processed_image_classification = load_preprocess_image_classification(image_np)
    processed_image_segmentation = load_and_prep_image_segmentation(image_np)
    
    # Classification
    classification_prediction = load_classification_model.predict(processed_image_classification)
    class_index = np.argmax(classification_prediction, axis=1)[0]
    class_label = classes[class_index]  # classes should be defined as ['Benign', 'Malignant', 'Normal']
    
    # Subtype classification
    subtype_label = 'Not applicable'
    if class_label == 'Benign':
        subtype_prediction = load_benign_subtype_model.predict(processed_image_classification)
        subtype_index = np.argmax(subtype_prediction, axis=1)[0]
        subtype_label = benign_subtype_mapping.get(subtype_index, 'Unknown subtype')
    elif class_label == 'Malignant':
        subtype_prediction = load_malignant_subtype_model.predict(processed_image_classification)
        subtype_index = np.argmax(subtype_prediction, axis=1)[0]
        subtype_label = malignant_subtype_mapping.get(subtype_index, 'Unknown subtype')
        
    # Segmentation
    segmentation_prediction = load_segmentation_model.predict(processed_image_segmentation)
    # Process the segmentation prediction if necessary
    
    # Grad-CAM
    # heatmap = make_gradcam_heatmap(processed_image_classification, classification_model, last_conv_layer_name)
    gradcam_img_array = preprocess_input(processed_image_classification)
    heatmap = make_gradcam_heatmap(gradcam_img_array, load_classification_model, 'conv5_block3_out')
    # Superimpose the heatmap on the original image
    
    # Combine results and send back
    results = {
        'classification': class_label,
        'subtype': subtype_label,
        'segmentation': segmentation_prediction.tolist(),  # Convert numpy array to list for JSON serialization
        'gradcam': heatmap.tolist()  # Convert numpy array to list for JSON serialization
    }
    print(results)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)