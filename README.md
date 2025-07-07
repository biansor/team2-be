# Face Comparison API

A high-accuracy face comparison REST API built with Flask and DeepFace for comparing selfie images. Perfect for identity verification and authentication systems.

## 🚀 Features

- **High Accuracy**: Uses DeepFace library with multiple AI models (VGG-Face, FaceNet, etc.)
- **Flexible Input**: Supports file uploads and base64 encoded images
- **Multiple Models**: Choose from 9 different face recognition models
- **RESTful API**: Clean and intuitive endpoints
- **Production Ready**: Error handling, file cleanup, and security measures
- **Real-time Processing**: Fast response times for immediate results

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Virtual environment (recommended)

## 🛠️ Installation

### 1. Clone or Download

```bash
git clone <your-repo-url>
cd face-comparison-api
```

### 2. Create Virtual Environment

```bash
python -m venv face_api_env
source face_api_env/bin/activate  # On Windows: face_api_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
python -m src.app
```

The API will start on `http://localhost:5000`

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
Flask==2.3.3
deepface==0.0.79
opencv-python==4.8.1.78
Pillow==10.0.1
numpy==1.24.3
Werkzeug==2.3.7
tf-keras==2.13.0
tensorflow==2.13.0
```

## 🎯 API Endpoints

### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "message": "Face comparison API is running"
}
```

### Compare Faces (File Upload)

```http
POST /compare
Content-Type: multipart/form-data
```

**Parameters:**

- `image1` (required): First image file
- `image2` (required): Second image file
- `model` (optional): AI model name (default: "VGG-Face")
- `distance_metric` (optional): Distance calculation method (default: "cosine")
- `threshold` (optional): Similarity threshold (default: 0.4)

**Success Response:**

```json
{
  "success": true,
  "verified": true,
  "distance": 0.23,
  "threshold": 0.4,
  "model": "VGG-Face",
  "distance_metric": "cosine",
  "similarity_percentage": 77.0
}
```

### Compare Faces (Base64)

```http
POST /compare-base64
Content-Type: application/json
```

**Request Body:**

```json
{
  "image1": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",
  "image2": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...",
  "model": "VGG-Face",
  "distance_metric": "cosine",
  "threshold": 0.4
}
```

### Get Available Models

```http
GET /models
```

**Response:**

```json
{
  "success": true,
  "models": [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace"
  ],
  "distance_metrics": ["cosine", "euclidean", "euclidean_l2"],
  "recommended_model": "VGG-Face",
  "recommended_distance_metric": "cosine"
}
```

## 🧪 Testing with Postman

### Quick Setup

1. **Import Collection**: Use the provided Postman collection JSON
2. **Set Base URL**: `http://localhost:5000`
3. **Test Health Check**: `GET /health`
4. **Upload Images**: Use `POST /compare` with form-data

### File Upload Test

1. **Method**: `POST`
2. **URL**: `http://localhost:5000/compare`
3. **Body**: Select "form-data"
4. **Add Fields**:
   - `image1`: (Type: File) - Select first image
   - `image2`: (Type: File) - Select second image

### Common Issues & Solutions

**Error: "Face comparison failed: 'distance_metric'"**

- Solution: Don't include optional parameters initially, or use exact values: `VGG-Face`, `cosine`, `0.4`

**Error: "No files selected"**

- Solution: Ensure you're selecting "File" type in Postman form-data, not "Text"

**Error: "Invalid file format"**

- Solution: Use supported formats: JPG, PNG, JPEG, GIF, BMP

## 🔧 Testing with cURL

### Basic Comparison

```bash
curl -X POST -F "image1=@selfie1.jpg" -F "image2=@selfie2.jpg" http://localhost:5000/compare
```

### With Parameters

```bash
curl -X POST \
  -F "image1=@selfie1.jpg" \
  -F "image2=@selfie2.jpg" \
  -F "model=VGG-Face" \
  -F "distance_metric=cosine" \
  -F "threshold=0.4" \
  http://localhost:5000/compare
```

### Health Check

```bash
curl -X GET http://localhost:5000/health
```

## 🎮 Testing with Python

```python
import requests

# File upload method
files = {
    'image1': open('selfie1.jpg', 'rb'),
    'image2': open('selfie2.jpg', 'rb')
}

response = requests.post('http://localhost:5000/compare', files=files)
print(response.json())

# Close files
files['image1'].close()
files['image2'].close()
```

## 🧠 Available Models

| Model          | Description                   | Speed     | Accuracy  |
| -------------- | ----------------------------- | --------- | --------- |
| **VGG-Face**   | ✅ Recommended - Best balance | Fast      | High      |
| **Facenet**    | High accuracy, Google's model | Medium    | Very High |
| **Facenet512** | Higher dimension version      | Slow      | Very High |
| **OpenFace**   | Lightweight, fast processing  | Very Fast | Good      |
| **DeepFace**   | Facebook's model              | Medium    | High      |
| **DeepID**     | Academic research model       | Medium    | Good      |
| **ArcFace**    | State-of-the-art accuracy     | Slow      | Very High |
| **Dlib**       | Traditional computer vision   | Fast      | Moderate  |
| **SFace**      | Balanced performance          | Medium    | High      |

## 📊 Distance Metrics

- **cosine** (✅ Recommended): Best for face comparison, range 0-1
- **euclidean**: Standard distance metric
- **euclidean_l2**: L2 normalized euclidean distance

## 🎚️ Threshold Guidelines

| Threshold | Use Case          | Description             |
| --------- | ----------------- | ----------------------- |
| 0.3       | Strict matching   | Very similar faces only |
| 0.4       | ✅ Recommended    | Balanced accuracy       |
| 0.6       | Moderate matching | More lenient            |
| 0.8       | Loose matching    | Very permissive         |

## 📁 Project Structure

```
face-comparison-api/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
├── temp_uploads/      # Temporary file storage (auto-created)
├── test_client.py     # Python test client
└── postman_collection.json  # Postman collection
```

## 🔒 Security Considerations

- **File Validation**: Only allows image file types
- **File Size Limits**: 16MB maximum file size
- **Temporary Files**: Automatic cleanup after processing
- **Input Sanitization**: Secure filename handling
- **Error Handling**: No sensitive information in error messages

## 🚀 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
EXPOSE 5000

CMD ["python", "app.py"]
```

### Environment Variables

```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
```

## 🎯 Performance Tips

1. **Use VGG-Face** for best speed/accuracy balance
2. **Optimize images** before upload (resize to reasonable dimensions)
3. **Use cosine distance** for face comparison
4. **Consider caching** for repeated comparisons
5. **Implement rate limiting** for production use

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Solution: Activate virtual environment
source face_api_env/bin/activate  # Linux/Mac
face_api_env\Scripts\activate     # Windows
```

**2. TensorFlow Warnings**

```bash
# Solution: Install specific version
pip install tensorflow==2.13.0
```

**3. OpenCV Issues**

```bash
# Solution: Try headless version
pip install opencv-python-headless
```

**4. Memory Issues**

- Reduce image sizes before processing
- Use lighter models (OpenFace, Dlib)
- Consider cloud deployment for better resources

### Debug Mode

Run with debug information:

```bash
export FLASK_DEBUG=True
python app.py
```

## 📈 Response Interpretation

### Success Response Fields

- `verified`: Boolean - Are the faces the same person?
- `distance`: Float - Similarity distance (lower = more similar)
- `threshold`: Float - Threshold used for comparison
- `similarity_percentage`: Float - Percentage similarity (0-100%)
- `model`: String - AI model used
- `distance_metric`: String - Distance calculation method

### Example Results

```json
{
  "verified": true, // Same person
  "distance": 0.23, // Very similar (low distance)
  "similarity_percentage": 77.0, // 77% similar
  "threshold": 0.4 // Used threshold
}
```
