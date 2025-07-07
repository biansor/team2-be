import os
import cv2
from src.config import MODELS, DEFAULT_THRESHOLD
import math

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def cleanup_file(filepath):
    print("""Remove temporary files""")
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up file {filepath}: {e}")

def preprocess_image(image_path):
    print("""Preprocess image to ensure it's in the right format""")
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False
    
def calculate_similarity_percentage(distance, metric, threshold):
    print("""Calculate similarity percentage based on distance metric""")

    if metric == 'cosine':
        _scale = threshold * 100
        if distance <= threshold:
            similarity = _scale + (_scale * (1 - (distance / threshold)))
        else:
            max_distance = min(1.0, threshold * 2) 
            similarity = max(0, 50 * (1 - ((distance - threshold) / (max_distance - threshold))))
    
    elif metric in ['euclidean', 'euclidean_l2', 'manhattan']:
        if distance <= threshold:
            similarity = 50 + (50 * (1 - (distance / threshold)))
        else:
            similarity = 50 * math.exp(-(distance - threshold) / threshold)
        
    else:
        if distance <= threshold:
            similarity = 50 + (50 * (1 - (distance / threshold)))
        else:
            max_distance = min(1.0, threshold * 2)
            similarity = max(0, 50 * (1 - ((distance - threshold) / (max_distance - threshold))))
    
    similarity = max(0, min(100, similarity))
    return round(similarity, 2)

def get_threshold(model, metric):
    print("Get threshold")
    try:
        threshold = MODELS[model][metric]
        print(f"Model: {model}, Metric: {metric}, Threshold: {threshold}")
        return threshold
    except KeyError as e:
        print(f"Error: {e}. Available models: {list(MODELS.keys())}. Metrics per model: {list(MODELS[model].keys()) if model in MODELS else 'N/A'}")
        return DEFAULT_THRESHOLD