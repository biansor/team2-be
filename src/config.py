import os
from dotenv import load_dotenv

load_dotenv()

# Flask configuration
PORT = int(os.getenv('PORT', 5001))
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'pjpeg', 'heic'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Face recognition backends
BACKENDS = [
    'retinaface', 'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'mediapipe', 'yolov8', 'yolov11s',
    'yolov11n', 'yolov11m', 'yunet', 'centerface',
]

# Model configurations
MODELS = {
    "VGG-Face": {
        "description": "VGG-Face refers to a series of deep convolutional neural network models developed by the Visual Geometry Group (VGG) at the University of Oxford for face recognition. They are trained on large-scale datasets like VGGFace2 and are known for their strong performance.",
        "cosine": 0.65,
        "euclidean": 1.70,
        "euclidean_l2": 1.70
    },
    "Facenet": {
        "description": "FaceNet is a facial recognition system developed by Google. It uses a deep convolutional neural network and a triplet loss function to directly learn a mapping from face images to a compact Euclidean embedding space, where distances directly correspond to face similarity.",
        "cosine": 0.64,
        "euclidean": 10,
        "euclidean_l2": 0.80
    },
    "Facenet512": {
        "description": "A variant of FaceNet that specifically produces 512-dimensional face embeddings. It often uses an Inception Residual Masking Network pre-trained on VGGFace2 to classify facial identities and provides this 512-dimensional latent facial embedding space.",
        "cosine": 0.55,
        "euclidean": 23.56,
        "euclidean_l2": 1.2
    },
    "ArcFace": {
        "description": "ArcFace is a deep learning model for face recognition that focuses on enhancing discriminative power by introducing an 'Additive Angular Margin Loss'. This loss function optimizes the feature embedding to enforce higher similarity for intra-class samples and greater diversity for inter-class samples, leading to highly discriminative features distributed on a hypersphere.",
        "cosine": 0.65,
        "euclidean": 4.15,
        "euclidean_l2": 1.13
    },
    "Dlib": {
        "description": "Dlib is a modern C++ toolkit that includes various machine learning algorithms, and it offers high-quality face recognition capabilities. Its face recognition model is often based on a ResNet-like architecture trained on a large dataset of faces, using a structured metric loss to project identities into distinct clusters.",
        "cosine": 0.07,
        "euclidean": 0.6,
        "euclidean_l2": 0.4
    },
    "SFace": {
        "description": "SFace is a lightweight face recognition model. While specific details on its architecture might vary or be less widely publicized than some other models, it's generally designed for efficient face recognition, often aiming for good performance with reduced computational overhead.",
        "cosine": 0.593,
        "euclidean": 10.734,
        "euclidean_l2": 1.055
    },
    "OpenFace": {
        "description": "OpenFace is an open-source project for face recognition and facial behavior analysis. OpenFace 3.0, for instance, offers a lightweight unified model for facial landmark detection, action unit detection, gaze estimation, and emotion recognition, utilizing a multi-task architecture for efficiency and performance.",
        "cosine": 0.583,
        "euclidean": 0.55,
        "euclidean_l2": 0.55
    },
    "DeepFace": {
        "description": "DeepFace is a deep learning facial recognition system developed by Facebook. It employs a nine-layer neural network with a significant number of parameters and was trained on a massive dataset of Facebook user images. It focuses on processes like 2D/3D alignment and frontalization before feeding the image to the neural network.",
        "cosine": 0.23,
        "euclidean": 64,
        "euclidean_l2": 0.64
    },
    "DeepID": {
        "description": "DeepID refers to a series of deep learning models that aim to learn high-level feature representations (Deep hidden IDentity features) for face verification. They are often trained on challenging multi-class face identification tasks, with a focus on learning compact and discriminative features even with varying poses.",
        "cosine": 0.15,
        "euclidean": 45,
        "euclidean_l2": 0.17
    },
    "GhostFaceNet": {
        "description": "GhostFaceNet is a lightweight face recognition model that combines the efficiency of GhostNet modules with the discriminative power of ArcFace loss. It's designed for high performance with minimal computational overhead, making it suitable for mobile and embedded devices by reducing feature map redundancy.",
        "cosine": 0.65,
        "euclidean": 35.71,
        "euclidean_l2": 1.10
    }
}

DISTANCE_METRICS = ['cosine', 'euclidean', 'euclidean_l2']
DEFAULT_MODEL = 'Facenet512'
DEFAULT_DISTANCE_METRIC = 'cosine'
DEFAULT_THRESHOLD = 0.5
DEFAULT_DETECTOR='retinaface'