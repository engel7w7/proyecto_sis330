import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOTIPO_ROOT = os.path.dirname(BASE_DIR)

DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_SAVED_DIR = os.path.join(BASE_DIR, 'src', 'models', 'saved')

DATASET_BASE_DIR = os.path.join(PROTOTIPO_ROOT, 'datasets')
DATASET_FACEFORENSICS_DIR = os.path.join(DATASET_BASE_DIR, 'FaceForensics')
DATASET_ASVSPOOF_2019_DIR = os.path.join(DATASET_BASE_DIR, 'ASVspoof_2019_Dataset')
DATASET_CIFAKE_DIR = os.path.join(DATASET_BASE_DIR, 'CIFAKE')

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_SAVED_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 224

AUDIO_MODEL_PTH = os.path.join(MODELS_SAVED_DIR, 'mejor_audio.pth')
VISION_MODEL_PTH = os.path.join(MODELS_SAVED_DIR, 'mejor_vision.pth')

AUDIO_TFLITE_PATH = os.path.join(MODELS_SAVED_DIR, 'experto_audio_int8.tflite')
VISION_TFLITE_PATH = os.path.join(MODELS_SAVED_DIR, 'experto_vision_int8.tflite')
