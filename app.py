import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()
port = int(os.getenv('PORT', 5001))

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(filepath):
    """Remove temporary files"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up file {filepath}: {e}")

def preprocess_image(image_path):
    """Preprocess image to ensure it's in the right format"""
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save back the processed image
        cv2.imwrite(image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Face comparison API is running'
    })

@app.route('/compare', methods=['POST'])
def compare_faces():
    print("""Compare two face images""")
    try:
        # Check if the request has files
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'error': 'Both image1 and image2 files are required',
                'success': False
            }), 400

        file1 = request.files['image1']
        file2 = request.files['image2']

        # Check if files are selected
        if file1.filename == '' or file2.filename == '':
            return jsonify({
                'error': 'No files selected',
                'success': False
            }), 400

        # Check file extensions
        if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
            return jsonify({
                'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, gif, bmp',
                'success': False
            }), 400

        # Save files temporarily
        filename1 = secure_filename(f"temp1_{file1.filename}")
        filename2 = secure_filename(f"temp2_{file2.filename}")
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        file1.save(filepath1)
        file2.save(filepath2)

        # Preprocess images
        if not preprocess_image(filepath1) or not preprocess_image(filepath2):
            cleanup_file(filepath1)
            cleanup_file(filepath2)
            return jsonify({
                'error': 'Failed to process images',
                'success': False
            }), 400

        # Get optional parameters with proper validation
        model_name = request.form.get('model', 'VGG-Face')  # Default model
        distance_metric = request.form.get('distance_metric', 'cosine')
        threshold = float(request.form.get('threshold', 0.4))  # Default threshold
        
        # Validate distance metric
        valid_metrics = ['cosine', 'euclidean', 'euclidean_l2']
        if distance_metric not in valid_metrics:
            distance_metric = 'cosine'

        # Perform face comparison
        result = DeepFace.verify(
            img1_path=filepath1,
            img2_path=filepath2,
            model_name=model_name,
            distance_metric=distance_metric,
            threshold=threshold
        )

        # Clean up temporary files
        cleanup_file(filepath1)
        cleanup_file(filepath2)

        # Return results
        response = {
            'success': True,
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'distance_metric': result.get('distance_metric', distance_metric),
            'similarity_percentage': round((1 - result['distance']) * 100, 2) if result['distance'] < 1 else 0
        }

        return jsonify(response)

    except Exception as e:
        # Clean up files in case of error
        cleanup_file(filepath1 if 'filepath1' in locals() else '')
        cleanup_file(filepath2 if 'filepath2' in locals() else '')
        
        return jsonify({
            'error': f'Face comparison failed: {str(e)}',
            'success': False
        }), 500

@app.route('/compare-base64', methods=['POST'])
def compare_faces_base64():
    """Compare two face images sent as base64 strings"""
    try:
        data = request.get_json()
        
        if not data or 'image1' not in data or 'image2' not in data:
            return jsonify({
                'error': 'Both image1 and image2 base64 strings are required',
                'success': False
            }), 400

        # Decode base64 images
        try:
            img1_data = base64.b64decode(data['image1'].split(',')[1] if ',' in data['image1'] else data['image1'])
            img2_data = base64.b64decode(data['image2'].split(',')[1] if ',' in data['image2'] else data['image2'])
        except Exception as e:
            return jsonify({
                'error': 'Invalid base64 image data',
                'success': False
            }), 400

        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
            tmp1.write(img1_data)
            filepath1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
            tmp2.write(img2_data)
            filepath2 = tmp2.name

        # Get optional parameters with proper validation
        model_name = data.get('model', 'VGG-Face')
        distance_metric = data.get('distance_metric', 'cosine')
        threshold = float(data.get('threshold', 0.4))
        
        # Validate distance metric
        valid_metrics = ['cosine', 'euclidean', 'euclidean_l2']
        if distance_metric not in valid_metrics:
            distance_metric = 'cosine'

        # Perform face comparison
        result = DeepFace.verify(
            img1_path=filepath1,
            img2_path=filepath2,
            model_name=model_name,
            distance_metric=distance_metric,
            threshold=threshold
        )

        # Clean up temporary files
        cleanup_file(filepath1)
        cleanup_file(filepath2)

        # Return results
        response = {
            'success': True,
            'verified': result['verified'],
            'distance': result['distance'],
            'threshold': result['threshold'],
            'model': result['model'],
            'distance_metric': result.get('distance_metric', distance_metric),
            'similarity_percentage': round((1 - result['distance']) * 100, 2) if result['distance'] < 1 else 0
        }

        return jsonify(response)

    except Exception as e:
        # Clean up files in case of error
        cleanup_file(filepath1 if 'filepath1' in locals() else '')
        cleanup_file(filepath2 if 'filepath2' in locals() else '')
        
        return jsonify({
            'error': f'Face comparison failed: {str(e)}',
            'success': False
        }), 500

@app.route('/models', methods=['GET'])
def get_available_models():
    """Get list of available face recognition models"""
    models = [
        'VGG-Face',
        'Facenet',
        'Facenet512',
        'OpenFace',
        'DeepFace',
        'DeepID',
        'ArcFace',
        'Dlib',
        'SFace'
    ]
    
    distance_metrics = ['cosine', 'euclidean', 'euclidean_l2']
    
    return jsonify({
        'success': True,
        'models': models,
        'distance_metrics': distance_metrics,
        'recommended_model': 'VGG-Face',
        'recommended_distance_metric': 'cosine'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Maximum size is 16MB',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    print("Face Comparison API starting...")
    print("Available endpoints:")
    print("  POST /compare - Compare two face images (multipart/form-data)")
    print("  POST /compare-base64 - Compare two face images (base64 JSON)")
    print("  GET /models - Get available models and distance metrics")
    print("  GET /health - Health check")
    
    app.run(debug=True, host='0.0.0.0', port=port)