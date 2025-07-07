import os
from flask import request, jsonify
from werkzeug.utils import secure_filename
from deepface import DeepFace
import tempfile
import base64
from src.config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MODELS, DISTANCE_METRICS, DEFAULT_MODEL, DEFAULT_DISTANCE_METRIC, DEFAULT_THRESHOLD, BACKENDS
from src.utils import allowed_file, cleanup_file, preprocess_image

def register_routes(app):
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'message': 'Face comparison API is running'
        })

    @app.route('/compare', methods=['POST'])
    def compare_faces():
        """Compare two face images"""
        try:
            if 'image1' not in request.files or 'image2' not in request.files:
                return jsonify({
                    'error': 'Both image1 and image2 files are required',
                    'success': False
                }), 400

            file1 = request.files['image1']
            file2 = request.files['image2']

            if file1.filename == '' or file2.filename == '':
                return jsonify({
                    'error': 'No files selected',
                    'success': False
                }), 400

            if not (allowed_file(file1.filename, ALLOWED_EXTENSIONS) and allowed_file(file2.filename, ALLOWED_EXTENSIONS)):
                return jsonify({
                    'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, gif, bmp',
                    'success': False
                }), 400

            filename1 = secure_filename(f"temp1_{file1.filename}")
            filename2 = secure_filename(f"temp2_{file2.filename}")
            filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
            filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

            file1.save(filepath1)
            file2.save(filepath2)

            if not preprocess_image(filepath1) or not preprocess_image(filepath2):
                cleanup_file(filepath1)
                cleanup_file(filepath2)
                return jsonify({
                    'error': 'Failed to process images',
                    'success': False
                }), 400

            model_name = request.form.get('model', DEFAULT_MODEL)
            distance_metric = request.form.get('distance_metric', DEFAULT_DISTANCE_METRIC)
            threshold = float(request.form.get('threshold', DEFAULT_THRESHOLD))
            detector_backend = request.form.get('detector_backend', BACKENDS[0])

            if distance_metric not in DISTANCE_METRICS:
                distance_metric = DEFAULT_DISTANCE_METRIC
            if detector_backend not in BACKENDS:
                detector_backend = BACKENDS[0]

            result = DeepFace.verify(
                img1_path=filepath1,
                img2_path=filepath2,
                model_name=model_name,
                distance_metric=distance_metric,
                threshold=threshold,
                detector_backend=detector_backend
            )

            cleanup_file(filepath1)
            cleanup_file(filepath2)

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

            try:
                img1_data = base64.b64decode(data['image1'].split(',')[1] if ',' in data['image1'] else data['image1'])
                img2_data = base64.b64decode(data['image2'].split(',')[1] if ',' in data['image2'] else data['image2'])
            except Exception as e:
                return jsonify({
                    'error': 'Invalid base64 image data',
                    'success': False
                }), 400

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp1:
                tmp1.write(img1_data)
                filepath1 = tmp1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp2:
                tmp2.write(img2_data)
                filepath2 = tmp2.name

            model_name = data.get('model', DEFAULT_MODEL)
            distance_metric = data.get('distance_metric', DEFAULT_DISTANCE_METRIC)
            threshold = float(data.get('threshold', DEFAULT_THRESHOLD))
            detector_backend = data.get('detector_backend', BACKENDS[0])

            if distance_metric not in DISTANCE_METRICS:
                distance_metric = DEFAULT_DISTANCE_METRIC
            if detector_backend not in BACKENDS:
                detector_backend = BACKENDS[0]

            result = DeepFace.verify(
                img1_path=filepath1,
                img2_path=filepath2,
                model_name=model_name,
                distance_metric=distance_metric,
                threshold=threshold,
                detector_backend=detector_backend
            )

            cleanup_file(filepath1)
            cleanup_file(filepath2)

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
            cleanup_file(filepath1 if 'filepath1' in locals() else '')
            cleanup_file(filepath2 if 'filepath2' in locals() else '')
            return jsonify({
                'error': f'Face comparison failed: {str(e)}',
                'success': False
            }), 500

    @app.route('/models', methods=['GET'])
    def get_available_models():
        """Get list of available face recognition models"""
        return jsonify({
            'success': True,
            'models': MODELS,
            'distance_metrics': DISTANCE_METRICS,
            'backends': BACKENDS,
            'recommended_model': DEFAULT_MODEL,
            'recommended_distance_metric': DEFAULT_DISTANCE_METRIC
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