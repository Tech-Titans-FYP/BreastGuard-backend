from flask import Flask, request, jsonify
from flask_cors import CORS
from routes.ultrasound import ultrasound_image_modality
from routes.mri import mri_image_modality
from routes.histo import histopathology_image_modality

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes
app.config['UPLOAD_FOLDER'] = 'path_to_upload_folder'

@app.route('/api/process-us-image', methods=['POST'])
def process_us_image():
    data = request.get_json()
    if 'image' in data and isinstance(data['image'], list) and len(data['image']) > 0:
        image_info = data['image'][0]
        if 'name' in image_info:
            filename = image_info['name']
            if 'mammogram' in filename.lower() or 'mri' in filename.lower() or 'histopathology' in filename.lower():
                return jsonify({'message': 'The submitted image could not be confidently classified as an valid image. Please upload a valid image!'}), 400
    return ultrasound_image_modality()

@app.route('/api/process-mri-image', methods=['POST'])
def process_mri_image():
    data = request.get_json()
    if 'image' in data and isinstance(data['image'], list) and len(data['image']) > 0:
        image_info = data['image'][0]
        if 'name' in image_info:
            filename = image_info['name']
            if 'mammogram' in filename.lower() or 'ultrasound' in filename.lower() or 'histopathology' in filename.lower():
                return jsonify({'message': 'The submitted image could not be confidently classified as an valid image. Please upload a valid image!'}), 400
    return mri_image_modality()

@app.route('/api/process-histo-image', methods=['POST'])
def process_histo_image():
    data = request.get_json()
    if 'image' in data and isinstance(data['image'], list) and len(data['image']) > 0:
        image_info = data['image'][0]
        if 'name' in image_info:
            filename = image_info['name']
            if 'mammogram' in filename.lower() or 'ultrasound' in filename.lower() or 'mri' in filename.lower():
                return jsonify({'message': 'The submitted image could not be confidently classified as an valid image. Please upload a valid image!'}), 400
    return histopathology_image_modality()

if __name__ == '__main__':
    app.run(debug=True)
