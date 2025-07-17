import os
from flask import request, jsonify
from werkzeug.utils import secure_filename
from deepface import DeepFace
import tempfile
import base64
from src.config import MODELS, DISTANCE_METRICS, DEFAULT_MODEL, DEFAULT_DISTANCE_METRIC, BACKENDS
from src.utils import is_image, cleanup_file, preprocess_image, calculate_similarity_percentage, get_threshold
import json

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

            if not (is_image(file1) and is_image(file2)):
                return jsonify({
                    'error': 'Invalid image file(s). Ensure both files are valid images.',
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

            _distance_metric =  get_threshold(model_name, distance_metric)
            
            threshold = float(request.form.get('threshold', _distance_metric))
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
            
            similarity_percentage = calculate_similarity_percentage(result['distance'], distance_metric, threshold)

            pretty_json = json.dumps(result, indent=4)
            print(pretty_json)

            response = {
                'success': True,
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'model': result['model'],
                'distance_metric': result.get('distance_metric', distance_metric),
                'similarity_percentage': similarity_percentage
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

            model_name = request.form.get('model', DEFAULT_MODEL)
            distance_metric = request.form.get('distance_metric', DEFAULT_DISTANCE_METRIC)

            _distance_metric =  get_threshold(model_name, distance_metric)
            
            threshold = float(request.form.get('threshold', _distance_metric))
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

            similarity_percentage = calculate_similarity_percentage(result['distance'], distance_metric, threshold)

            response = {
                'success': True,
                'verified': result['verified'],
                'distance': result['distance'],
                'threshold': result['threshold'],
                'model': result['model'],
                'distance_metric': result.get('distance_metric', distance_metric),
                'similarity_percentage': similarity_percentage
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

    @app.route('/analyze', methods=['POST'])
    def analyze_image():
        """Analyze face image for age, gender, emotion, and race"""
        print("Analyzing photo")
        try:
            if 'image' not in request.files:
                return jsonify({
                    'error': 'Image file is required',
                    'success': False
                }), 400

            file = request.files['image']

            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'success': False
                }), 400


            if not (is_image(file)):
                return jsonify({
                    'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, gif, bmp',
                    'success': False
                }), 400

            filename = secure_filename(f"analyze_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            if not preprocess_image(filepath):
                cleanup_file(filepath)
                return jsonify({
                    'error': 'Failed to process image',
                    'success': False
                }), 400

            # Get optional parameters from form data
            detector_backend = request.form.get('detector_backend', BACKENDS[0])
            actions = request.form.get('actions', 'age,gender,emotion,race').split(',')
            
            # Validate detector backend
            if detector_backend not in BACKENDS:
                detector_backend = BACKENDS[0]
                
            # Validate actions
            valid_actions = ['age', 'gender', 'emotion', 'race']
            actions = [action.strip() for action in actions if action.strip() in valid_actions]
            
            if not actions:
                actions = ['age', 'gender', 'emotion', 'race']  # Default to all

            # Perform analysis
            analysis_result = DeepFace.analyze(
                img_path=filepath,
                actions=actions,
                detector_backend=detector_backend,
                enforce_detection=True
            )

            cleanup_file(filepath)

            def convert_numpy_types(obj):
                """Convert numpy types to Python native types for JSON serialization"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj

            # Handle both single face and multiple faces results
            if isinstance(analysis_result, list):
                # Multiple faces detected
                faces_analysis = []
                for i, face_data in enumerate(analysis_result):
                    face_info = {
                        'face_index': i,
                        'region': convert_numpy_types(face_data.get('region', {})),
                    }
                    
                    # Add analysis results based on requested actions
                    if 'age' in actions:
                        face_info['age'] = convert_numpy_types(face_data.get('age', 0))
                        
                    if 'gender' in actions:
                        gender_data = face_data.get('gender', {})
                        face_info['gender'] = {
                            'dominant': str(face_data.get('dominant_gender', 'unknown')),
                            'probabilities': convert_numpy_types(gender_data)
                        }
                        
                    if 'emotion' in actions:
                        emotion_data = face_data.get('emotion', {})
                        face_info['emotion'] = {
                            'dominant': str(face_data.get('dominant_emotion', 'unknown')),
                            'probabilities': convert_numpy_types(emotion_data)
                        }
                        
                    if 'race' in actions:
                        race_data = face_data.get('race', {})
                        face_info['race'] = {
                            'dominant': str(face_data.get('dominant_race', 'unknown')),
                            'probabilities': convert_numpy_types(race_data)
                        }
                    
                    faces_analysis.append(face_info)
                    
                analysis_summary = {
                    'faces_detected': len(analysis_result),
                    'faces': faces_analysis
                }
            else:
                # Single face detected
                face_info = {
                    'region': convert_numpy_types(analysis_result.get('region', {})),
                }
                
                # Add analysis results based on requested actions
                if 'age' in actions:
                    face_info['age'] = convert_numpy_types(analysis_result.get('age', 0))
                    
                if 'gender' in actions:
                    gender_data = analysis_result.get('gender', {})
                    face_info['gender'] = {
                        'dominant': str(analysis_result.get('dominant_gender', 'unknown')),
                        'probabilities': convert_numpy_types(gender_data)
                    }
                    
                if 'emotion' in actions:
                    emotion_data = analysis_result.get('emotion', {})
                    face_info['emotion'] = {
                        'dominant': str(analysis_result.get('dominant_emotion', 'unknown')),
                        'probabilities': convert_numpy_types(emotion_data)
                    }
                    
                if 'race' in actions:
                    race_data = analysis_result.get('race', {})
                    face_info['race'] = {
                        'dominant': str(analysis_result.get('dominant_race', 'unknown')),
                        'probabilities': convert_numpy_types(race_data)
                    }
                    
                analysis_summary = {
                    'faces_detected': 1,
                    'faces': [face_info]
                }

            pretty_json = json.dumps(analysis_summary, indent=4)
            print(pretty_json)

            response = {
                'success': True,
                'detector_backend': detector_backend,
                'actions_performed': actions,
                **analysis_summary
            }

            return jsonify(response)

        except Exception as e:
            cleanup_file(filepath if 'filepath' in locals() else '')
            return jsonify({
                'error': f'Image analysis failed: {str(e)}',
                'success': False
            }), 500

    @app.route('/liveness-check', methods=['POST'])
    def check_liveness():
        """Check if face image is live or spoofed"""
        try:
            if 'image' not in request.files:
                return jsonify({
                    'error': 'Image file is required',
                    'success': False
                }), 400

            file = request.files['image']

            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'success': False
                }), 400

            if not (is_image(file)):
                return jsonify({
                    'error': 'Invalid file format. Allowed formats: png, jpg, jpeg, gif, bmp',
                    'success': False
                }), 400

            filename = secure_filename(f"liveness_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            if not preprocess_image(filepath):
                cleanup_file(filepath)
                return jsonify({
                    'error': 'Failed to process image',
                    'success': False
                }), 400

            # Params
            detector_backend = request.form.get('detector_backend', BACKENDS[0])
            enforce_detection = request.form.get('enforce_detection', False)
            
            if detector_backend not in BACKENDS:
                detector_backend = BACKENDS[0]

            # Extract faces with anti-spoofing
            faces = DeepFace.extract_faces(
                img_path=filepath,
                anti_spoofing=True,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection
            )

            cleanup_file(filepath)

            if not faces:
                return jsonify({
                    'success': False,
                    'is_live': False,
                    'confidence': 0.0,
                    'faces_detected': 0,
                    'message': 'No face detected'
                })

            # Get the first face result
            face_result = faces[0]

            print(faces)
            
            # Check if the face appears to be live
            # Note: The exact structure may vary depending on DeepFace version
            is_live = face_result.get('is_real', True)  # Default to True if not available
            confidence = face_result.get('antispoof_score', 1.0)  # Default confidence

            pretty_json = json.dumps({
                'is_live': is_live,
                'confidence': float(confidence),
                'faces_detected': len(faces),
                'detector_backend': detector_backend
            }, indent=4)
            print(pretty_json)

            response = {
                'success': True,
                'is_live': is_live,
                'confidence': float(confidence),
                'faces_detected': len(faces),
                'detector_backend': detector_backend,
                'message': 'Live face detected' if is_live else 'Spoofed face detected'
            }

            return jsonify(response)

        except Exception as e:
            cleanup_file(filepath if 'filepath' in locals() else '')
            return jsonify({
                'error': f'Liveness check failed: {str(e)}',
                'success': False
            }), 500

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