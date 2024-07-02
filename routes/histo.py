

from flask import request, jsonify
import base64
import numpy as np
from utils.histo_util import preprocess_for_prediction, get_gradcam_heatmap, plot_gradcam
from models.histo_model import load_classification_model, load_subtype_model

classification_model = load_classification_model()
subtype_model = load_subtype_model()

subtype_descriptions = {
    'Malignant Ductal Carcinoma': {
        'description': 'Malignant ductal carcinoma, also known as invasive ductal carcinoma (IDC), is the most common type of breast cancer...',
        'features': [
            'Pleomorphism (variation in cell size and shape).',
            'Increased mitotic activity (indicating rapid cell division).',
            'Irregular glandular structures or solid nests.',
            'Necrosis (areas of dead cells) and stromal desmoplasia (fibrous tissue response).'
        ],
        'guidance': [
            'Diagnosis: Often detected through mammograms...',
            'Treatment: Surgery (lumpectomy or mastectomy)...',
            'Prognosis: Depends on the stage at diagnosis...',
            'Support: Joining a support group...'
        ]
    },
    'Benign Fibroadenoma': {
        'description': 'Fibroadenoma is a common benign breast tumor...',
        'features': [
            'Mixed proliferation of glandular and stromal components.',
            'Compressed and distorted glandular elements.',
            'Fibroblastic stroma that can range from myxoid to sclerotic.'
        ],
        'guidance': [
            'Diagnosis: Physical exams, ultrasound...',
            'Treatment: Often monitored without intervention...',
            'Prognosis: Excellent, with very low risk...',
            'Support: Regular self-exams...'
        ]
    },
    # ... other subtype descriptions ...
}

benign_subtype_mapping = {
    0: 'Benign Fibroadenoma',
    1: 'Benign Phyllodes Tumor',
    2: 'Benign Adenosis',
    3: 'Benign Tubular Adenoma'
}

malignant_subtype_mapping = {
    0: 'Malignant Ductal Carcinoma',
    1: 'Malignant Lobular Carcinoma',
    2: 'Malignant Mucinous Carcinoma',
    3: 'Malignant Papillary Carcinoma'
}

def histopathology_image_modality():
    data = request.get_json()
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
        image_data = preprocess_for_prediction(image_data)
    except (base64.binascii.Error, UnidentifiedImageError):
        return jsonify({'message': 'Invalid image data'}), 400
    
    classification_prediction = classification_model.predict(image_data)
    predicted_class = 'malignant' if classification_prediction[0] >= 0.5 else 'benign'
    confidence = float(np.max(classification_prediction))
    
    subtype_info = {}
    if predicted_class == 'malignant':
        subtype_prediction = subtype_model.predict(image_data)
        subtype_index = np.argmax(subtype_prediction)
        subtype_class = malignant_subtype_mapping.get(subtype_index, "Unknown")
        subtype_confidence = float(np.max(subtype_prediction))
        if subtype_class in subtype_descriptions:
            subtype_info = {
                'subtype': subtype_class,
                'subtype_confidence': subtype_confidence,
                'description': subtype_descriptions[subtype_class]['description'],
                'features': subtype_descriptions[subtype_class]['features'],
                'guidance': subtype_descriptions[subtype_class]['guidance']
            }
        else:
            subtype_info = {
                'subtype': "Unknown",
                'subtype_confidence': subtype_confidence,
                'description': "No description available.",
                'features': [],
                'guidance': []
            }
    else:
        subtype_prediction = subtype_model.predict(image_data)
        subtype_index = np.argmax(subtype_prediction)
        subtype_class = benign_subtype_mapping.get(subtype_index, "Unknown")
        subtype_confidence = float(np.max(subtype_prediction))
        if subtype_class in subtype_descriptions:
            subtype_info = {
                'subtype': subtype_class,
                'subtype_confidence': subtype_confidence,
                'description': subtype_descriptions[subtype_class]['description'],
                'features': subtype_descriptions[subtype_class]['features'],
                'guidance': subtype_descriptions[subtype_class]['guidance']
            }
        else:
            subtype_info = {
                'subtype': "Unknown",
                'subtype_confidence': subtype_confidence,
                'description': "No description available.",
                'features': [],
                'guidance': []
            }

    heatmap = get_gradcam_heatmap(classification_model, image_data, 'conv2d')
    gradcam_image = plot_gradcam(np.squeeze(image_data), heatmap)

    results = {
        'classification': predicted_class,
        'subtype': subtype_class,
        'subtype_description': subtype_descriptions[subtype_class]['description'],
        'features': subtype_descriptions[subtype_class]['features'],
        'guidance': subtype_descriptions[subtype_class]['guidance'],
        'gradcam': gradcam_image
    }
    print(results)
    return jsonify(results)

