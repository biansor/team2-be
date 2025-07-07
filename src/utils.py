import os
import cv2

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

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
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        return True
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return False