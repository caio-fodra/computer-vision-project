<div align="center">

# Hand Sign Recognition

Real-time hand sign recognition with OpenCV, MediaPipe Hands, and a custom landmark classifier.

[Setup](#setup) | [Run the pipeline](#run-the-pipeline) | [Controls](#controls) | [Project layout](#project-layout)

</div>

---

## Overview

The application tracks 21 hand landmarks, converts them into 63 normalized features, and classifies signs with a scikit-learn MLP model. You can collect examples with your webcam and train a model for your setup. If no trained model is available, the app falls back to a small rule-based recognizer.

## Setup

Requirements: Python 3.9 or newer and a working webcam.

```bash
git clone https://github.com/caio-fodra/computer-vision-project.git
cd computer-vision-project
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the pipeline

### 1. Collect examples

```bash
python collect_data.py
```

Show a hand sign and press its letter key (`A`-`Z`). Each recording burst captures up to 150 frames. Collect at least two different letters, with at least 50 examples for each class you want to train.

### 2. Train the classifier

```bash
python train_landmarks.py --data data/landmarks.csv
```

The script reports evaluation metrics and saves the trained model to `models/landmark_model.joblib`.

### 3. Start recognition

```bash
python main.py
```

The application searches camera indexes for a working webcam and loads the trained model when available. Dataset files and trained models are generated locally; they are not required in the source commit.

## Controls

| Action | Control |
| --- | --- |
| Add a recognized sign | Hold the hand pose steady |
| Insert a space | Pinch thumb and index finger, or press `Space` |
| Delete the last character | `Backspace` |
| Clear the text | Show two hands |
| Exit | `Q` or `Esc` |

The app waits for a stable pose before adding a character. Release the pose between signs.

## Configuration

Adjust camera detection and tracking thresholds, model confidence, and data/model paths in `config.py`.

## Project layout

```text
.
|-- camera.py           # Webcam access and MediaPipe hand detection
|-- collect_data.py     # Webcam dataset collection
|-- config.py           # Paths, thresholds, and keyboard controls
|-- hand_features.py    # Landmark feature extraction and normalization
|-- main.py             # Real-time recognition and text input
|-- train_landmarks.py  # MLP training and evaluation
`-- requirements.txt    # Python dependencies
```

The `data/` and `models/` directories are created as needed and contain generated local files.

## Troubleshooting

- **No camera found:** Check that the webcam is connected and not being used by another application.
- **Model not found:** Run the collection and training steps above. The app uses heuristic recognition until a trained model is available.
- **Low recognition accuracy:** Collect more varied examples across hand angles, distances, and lighting conditions, then retrain.