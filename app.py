# app.py

from flask import Flask
from flask_cors import CORS
from routes.ultrasound import ultrasound_image_modality
from routes.mri import mri_image_modality
from routes.histo import histopathology_image_modality

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes
app.config['UPLOAD_FOLDER'] = 'path_to_upload_folder'

@app.route('/api/process-us-image', methods=['POST'])
def process_us_image():
    return ultrasound_image_modality()

@app.route('/api/process-mri-image', methods=['POST'])
def process_mri_image():
    return mri_image_modality()

@app.route('/api/process-histo-image', methods=['POST'])
def process_histo_image():
    return histopathology_image_modality()

if __name__ == '__main__':
    app.run(debug=True)
