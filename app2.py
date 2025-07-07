from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/compare-selfies', methods=['POST'])
def compare_selfies():
    try:
        # Check if the post request has the files
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Two images are required'}), 400

        image1 = request.files['image1']
        image2 = request.files['image2']

        # Validate file extensions
        if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, or JPEG'}), 400

        # Save the uploaded images temporarily
        image1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image1.filename))
        image2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image2.filename))
        image1.save(image1_path)
        image2.save(image2_path)

        # Perform face verification using DeepFace
        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=image2_path,
            model_name='ArcFace',  # Use ArcFace for high accuracy
            detector_backend='retinaface',  # RetinaFace for robust detection
            enforce_detection=True,  # Ensure faces are detected
            align=True  # Align faces for better accuracy
        )

        # Clean up the saved images
        os.remove(image1_path)
        os.remove(image2_path)

        # Prepare the response
        response = {
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'message': 'The selfies depict the same person' if result['verified'] else 'The selfies depict different people'
        }

        return jsonify(response), 200

    except Exception as e:
        # Clean up files in case of error
        if os.path.exists(image1_path):
            os.remove(image1_path)
        if os.path.exists(image2_path):
            os.remove(image2_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)