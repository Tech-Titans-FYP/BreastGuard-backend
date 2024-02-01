from flask import Flask
from flask_cors import CORS
from routes.ultrasound import ultrasound_image_modality
from routes.mri import mri_image_modality

app = Flask(__name__)
CORS(app) # Allow CORS for all routes
app.config['UPLOAD_FOLDER'] = 'E:\\L4S1\\IS4910 - Comprehensive Group Project\\FlaskDemo\\uploads'

@app.route('/api/process-us-image', methods=['POST'])
def process_us_image():
    return ultrasound_image_modality()

@app.route('/api/process-mri-image', methods=['POST'])
def process_mri_image():
    return mri_image_modality()

if __name__ == '__main__':
    app.run(debug=True)