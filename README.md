# EngageAI (Real-Time-Emotion-Detection)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-green.svg)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow.svg)](https://developer.chrome.com/docs/extensions/)

**Real-Time-Emotion-Detection** is an AI-driven, real-time facial emotion detection and engagement analysis tool. It utilizes deep learning and computer vision to assess user engagement via webcam or browser-based interfaces.

---

## 🌍 Overview

This project captures real-time facial expressions and processes them to detect seven distinct emotions. It then uses these emotional cues, along with head movement, gaze direction, and smile detection, to determine an overall engagement score.

![Emotion Detection Demo](https://via.placeholder.com/640x360.png?text=Emotion+Detection+Demo)

---

## ✨ Features

- 😊 **Emotion Recognition**: Detects 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
- 🧠 **Engagement Scoring**: Combines emotion, smile, gaze, and head movement into a final score
- 😎 **Real-Time Analysis**: Processes webcam video live with MediaPipe and OpenCV
- 🌐 **Flask API**: Enables easy integration with web and mobile apps
- 📲 **Chrome Extension**: Allows engagement monitoring during online meetings
- 🎓 **Training Support**: Includes training and testing scripts for custom models

---

## 🧠 Emotions & Scoring

| Emotion     | Score |
|-------------|-------|
| Angry       | 0.6   |
| Disgusted   | 0.1   |
| Fearful     | 0.4   |
| Happy       | 1.0   |
| Neutral     | 0.5 / 0.8 |
| Sad         | 0.2   |
| Surprised   | 0.8 / 0.9 |

---

## 📚 Technologies Used

- **Python** – Core language
- **TensorFlow / Keras** – Deep learning model for emotion classification
- **OpenCV** – Video and image processing
- **MediaPipe** – Facial landmark detection
- **Flask** – Backend API framework
- **NumPy** – Numerical computing
- **Chrome Extensions API** – Browser integration

---

## ⚙️ Installation

### Backend Setup
```bash
git clone https://github.com/yadavadarsh55/Real-Time-Emotion-Detection.git
cd Real-Time-Emotion-Detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Chrome Extension Setup
1. Go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and choose the `extension/` folder

---

## 🔧 Usage

### Standalone Webcam Mode
```bash
python main.py
```

### Flask API Server
```bash
python server.py
```

- `POST /analyze` – Analyze a base64 image
- `GET /status` – Server health check

### Chrome Extension (EngageAI)

- Start server (`python server.py`)
- Click the extension icon
- Press "Activate"
- Join a video meeting (Meet, Zoom, etc.)

Displays:
- Detected emotion
- Emotion score
- Engagement level

---

## 📄 API Reference

### POST `/analyze`
Analyze facial features from a base64 image.

**Request:**
```json
{
  "frame": "base64-encoded-image"
}
```
**Response:**
```json
{
  "faces": [
    {
      "emotion": "Happy",
      "emotion_score": 1.0,
      "smile_score": 1.0,
      "head_score": 0.6,
      "gaze_score": 1.0,
      "engagement_score": 0.88,
      "engagement_level": "Fully Engaged",
      "bounding_box": {
        "x": 120,
        "y": 80,
        "width": 200,
        "height": 200
      }
    }
  ]
}
```

### GET `/status`
Returns server and model health.

---

## 📈 Training Your Own Model

Prepare your dataset:
```
data/
├── train/
├── validation/
├── test/
    └── Angry/, Happy/, etc.
```
Train:
```bash
python train_model.py
```
Test:
```bash
python test_model.py
```

---

## 📊 Engagement Metrics

| Metric         | Weight |
|----------------|--------|
| Emotion Score  | 30%    |
| Smile Score    | 10%    |
| Head Position  | 40%    |
| Gaze Direction | 20%    |

**Levels:**
- Fully Engaged ≥ 60%
- Partially Engaged: 50–59.9%
- Not Engaged < 50%

---

## 🗂️ Project Structure

```
Real-Time-Emotion-Detection/
├── model/                 # Pretrained models
├── extension/             # Chrome Extension
├── emotion_score.py       # Emotion utils
├── main.py                # Webcam detection
├── server.py              # Flask API
├── train_model.py         # Training
├── test_model.py          # Evaluation
├── LICENSE.md
└── README.md
```

---

## 👜 Chrome Extension Breakdown

- `background.js` – Background scripts
- `content.js` – Injected scripts
- `popup.html` – User UI
- `popup.js` – Functionality
- Requires: `activeTab`, `storage`, `nativeMessaging`, `scripting`

---

## 👤 Contributors

- **Adarsh Yadav** – [@yadavadarsh55](https://github.com/yadavadarsh55)

---

## 🙏 Acknowledgments

- FER2013 dataset
- MediaPipe by Google
- TensorFlow team
- Chrome Extension APIs

---

## 📄 License

MIT License – see [LICENSE.md](LICENSE.md)

