from flask import Flask
from flask_cors import CORS
import os
from src.config import UPLOAD_FOLDER, MAX_CONTENT_LENGTH, PORT
from src.routes import register_routes

app = Flask(__name__)
CORS(app, origins="*")

# Apply configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Register routes
register_routes(app)

if __name__ == '__main__':
    print("Face Comparison API starting...")
    print("Available endpoints:")
    print(" POST /compare - Compare two face images (multipart/form-data)")
    print(" POST /compare-base64 - Compare two face images (base64 JSON)")
    print(" GET /models - Get available models and distance metrics")
    print(" GET /health - Health check")
    app.run(debug=True, host='0.0.0.0', port=PORT)