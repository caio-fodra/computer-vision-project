from pathlib import Path
 
DATA_PATH = Path("data") / "landmarks.csv"
MNIST_DATA_PATH = Path("data") / "mnist_landmarks.csv"
MODEL_PATH = Path("models") / "landmark_model.joblib"
 
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5
MIN_CONFIDENCE = 0.80
 
KEY_ESC = 27
KEY_BACKSPACE = 8
KEY_SPACE = 32