

from flask import request, jsonify
import base64
import numpy as np
from utils.histo_util import preprocess_for_prediction, get_gradcam_heatmap, plot_gradcam
from models.histo_model import load_classification_model, load_subtype_model

classification_model = load_classification_model()
subtype_model = load_subtype_model()

subtype_descriptions = {
    'Malignant Ductal Carcinoma': {
        'description': 'Malignant ductal carcinoma, also known as invasive ductal carcinoma (IDC), is the most common type of breast cancer, accounting for about 80% of all cases. Histologically, it originates in the milk ducts and invades the surrounding breast tissue.',
        'features': [
            'Pleomorphism (variation in cell size and shape).',
            'Increased mitotic activity (indicating rapid cell division).',
            'Irregular glandular structures or solid nests.',
            'Necrosis (areas of dead cells) and stromal desmoplasia (fibrous tissue response).'
        ],
        'guidance': [
            'Diagnosis: Often detected through mammograms, ultrasounds, or biopsies. Pathology confirms the diagnosis.',
            'Treatment: Surgery (lumpectomy or mastectomy), radiation, chemotherapy, hormone therapy, and targeted therapy.',
            'Prognosis: Depends on the stage at diagnosis, tumor size, and spread. Early detection improves outcomes.',
            'Support: Joining a support group and regular follow-ups with the oncology team can help in managing treatment and emotional well-being.'
        ]
    },
    'Benign Fibroadenoma': {
        'description': 'Fibroadenoma is a common benign breast tumor, often found in women under 30. It is characterized by a well-circumscribed, encapsulated lesion that feels like a firm, smooth, rubbery lump.',
        'features': [
            'Mixed proliferation of glandular and stromal components.',
            'Compressed and distorted glandular elements.',
            'Fibroblastic stroma that can range from myxoid to sclerotic.'
        ],
        'guidance': [
            'Diagnosis: Physical exams, ultrasound, and sometimes biopsy.',
            'Treatment: Often monitored without intervention. Surgical removal if symptomatic or for patient peace of mind.',
            'Prognosis: Excellent, with very low risk of becoming cancerous.',
            'Support: Regular self-exams and routine clinical check-ups are recommended.'
        ]
    },
    'Malignant Mucinous Carcinoma': {
        'description': 'Mucinous carcinoma, also known as colloid carcinoma, is a rare type of invasive breast cancer characterized by large amounts of extracellular mucin. It typically affects older women and grows slowly.',
        'features': [
            'Tumor cells floating in large pools of mucin.',
            'Well-differentiated cells forming clusters or single cells within the mucin.'
        ],
        'guidance': [
            'Diagnosis: Detected through mammography, ultrasound, and confirmed by biopsy.',
            'Treatment: Surgery (often with clear margins), radiation, and sometimes hormone therapy.',
            'Prognosis: Generally favorable due to slow growth and lower tendency to metastasize.',
            'Support: Regular follow-ups and monitoring for recurrence, with support from oncology nurses and social workers.'
        ]
    },
    'Malignant Lobular Carcinoma': {
        'description': 'Malignant lobular carcinoma, or invasive lobular carcinoma (ILC), originates in the lobules of the breast. It is the second most common type of invasive breast cancer.',
        'features': [
            'Small, uniform, discohesive cells.',
            'Single-file infiltration pattern.',
            'Signet-ring cells with intracytoplasmic mucin vacuoles.'
        ],
        'guidance': [
            'Diagnosis: Often harder to detect via mammogram; MRI or biopsy can be more effective.',
            'Treatment: Surgery, radiation, chemotherapy, hormone therapy, and targeted therapy.',
            'Prognosis: Depends on stage at diagnosis, with early-stage detection having a good prognosis.',
            'Support: Genetic counseling and support groups for patients with a family history of breast cancer.'
        ]
    },
    'Benign Tubular Adenoma': {
        'description': 'Tubular adenoma is a rare benign breast lesion, usually well-circumscribed and small. It is more common in younger women.',
        'features': [
            'Proliferation of small, round, or oval tubules.',
            'Lined by a single layer of epithelial cells.',
            'Minimal cellular atypia and low mitotic activity.'
        ],
        'guidance': [
            'Diagnosis: Detected through physical examination, imaging, and biopsy.',
            'Treatment: Often monitored. Surgical removal if symptomatic or for reassurance.',
            'Prognosis: Excellent, with no risk of cancer.',
            'Support: Regular self-exams and discussions with a healthcare provider about any changes.'
        ]
    },
    'Malignant Papillary Carcinoma': {
        'description': 'Papillary carcinoma is a rare malignant breast tumor characterized by the formation of papillary structures. It can be invasive or non-invasive (in situ).',
        'features': [
            'Papillary structures with fibrovascular cores.',
            'Malignant epithelial cells lining the papillae.',
            'Possible necrosis.'
        ],
        'guidance': [
            'Diagnosis: Detected via imaging and confirmed by biopsy.',
            'Treatment: Surgery, often followed by radiation. May require hormone or targeted therapy.',
            'Prognosis: Generally good, especially if detected early.',
            'Support: Emotional support through counseling and patient support groups.'
        ]
    },
    'Benign Phyllodes Tumor': {
        'description': 'Phyllodes tumor is a rare fibroepithelial tumor that can be benign, borderline, or malignant. Often presents as a fast-growing, painless lump.',
        'features': [
            'Biphasic pattern with stromal and epithelial components.',
            'Leaf-like projections.',
            'Hypercellular stroma without significant atypia.'
        ],
        'guidance': [
            'Diagnosis: Detected through physical exam, imaging, and biopsy.',
            'Treatment: Surgical excision with clear margins to prevent recurrence.',
            'Prognosis: Benign tumors have an excellent prognosis but require monitoring for recurrence.',
            'Support: Regular follow-ups and patient education about recurrence risks.'
        ]
    },
    'Benign Adenosis': {
        'description': 'Adenosis is a benign proliferative condition where the lobules of the breast are enlarged with more glands than usual.',
        'features': [
            'Expanded lobules with increased acini.',
            'Lined by epithelial and myoepithelial cells.',
            'No significant cellular atypia or increased mitotic activity.'
        ],
        'guidance': [
            'Diagnosis: Often found incidentally on mammograms or during evaluation of breast lumps.',
            'Treatment: Usually no treatment needed, but regular monitoring is recommended.',
            'Prognosis: Excellent, as it is a benign condition with no risk of cancer.',
            'Support: Routine mammograms and self-exams to monitor any changes.'
        ]
    }
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
        'subtype_description': subtype_info,
        'gradcam': gradcam_image
    }
    return jsonify(results)
