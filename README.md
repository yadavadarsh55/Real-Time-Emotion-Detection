# Real-Time-Emotion-Detection

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-green.svg)](https://mediapipe.dev/)
[![Flask](https://img.shields.io/badge/Flask-2.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow.svg)](https://developer.chrome.com/docs/extensions/)

A real-time facial emotion detection system with engagement analysis using deep learning and computer vision techniques, accessible through both a standalone application and a Chrome extension.

## Overview

Real-Time-Emotion-Detection detects and analyzes facial emotions to determine user engagement levels in real-time. The system processes video input to recognize seven distinct emotions and calculates an overall engagement score based on multiple facial features including emotion, smile detection, head position, and gaze direction.

![Emotion Detection Demo](https://via.placeholder.com/640x360.png?text=Emotion+Detection+Demo)

## Features

- **Real-time emotion recognition**: Detects 7 emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised)
- **Engagement analysis**: Calculates user engagement levels based on facial features
- **Head position tracking**: Detects head tilts and movements
- **Gaze detection**: Monitors eye gaze direction
- **Smile detection**: Recognizes smile expressions
- **Flask API server**: Backend service for web application integration
- **Chrome extension**: Browser-based emotion and engagement analysis for video meetings

## Emotions Detected

- Angry (Score: 0.6)
- Disgusted (Score: 0.1)
- Fearful (Score: 0.4)
- Happy (Score: 1.0)
- Neutral (Score: 0.5/0.8)
- Sad (Score: 0.2)
- Surprised (Score: 0.8/0.9)

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for emotion classification
- **OpenCV**: Computer vision library for image processing
- **MediaPipe**: Face mesh and landmark detection
- **Flask**: Web server for API endpoints
- **NumPy**: Numerical operations and array manipulation
- **Chrome Extensions API**: Browser integration for video conferencing platforms

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yadavadarsh55/Real-Time-Emotion-Detection.git
   cd Real-Time-Emotion-Detection
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model or train your own (see Training section).

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in the top right)
3. Click "Load unpacked" and select the `extension` folder from the repository
4. The "EngageAI" extension should now appear in your extensions list

## Usage

### Standalone Application

Launch the real-time emotion detection with your webcam:

```bash
python main.py
```

This will open a window showing the video feed with emotion detection and engagement analysis overlaid.

### API Server

Start the Flask server for web application and Chrome extension integration:

```bash
python server.py
```

The server will be available at `http://localhost:5000` with the following endpoints:
- `POST /analyze`: Analyze a frame for emotions and engagement
- `GET /status`: Check server status and model availability

### Chrome Extension (EngageAI)

The EngageAI extension is designed to work with popular video conferencing platforms:
- Google Meet
- Zoom
- Microsoft Teams
- Webex

To use the extension:

1. Make sure the Python server is running (`python server.py`)
2. Click the EngageAI extension icon in your browser toolbar
3. Click "Activate" in the popup
4. Join your video meeting, and the extension will automatically analyze engagement

The extension will display:
- Current detected emotion
- Emotion score
- Engagement level (Fully Engaged, Partially Engaged, Not Engaged)

## API Reference

### POST /analyze

Analyzes a single image frame for emotions and engagement metrics.

**Request Body:**
```json
{
  "frame": "base64-encoded-image-data"
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

### GET /status

Checks if the server and model are running properly.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "analyzer_ready": true
}
```

## Training

To train your own emotion recognition model:

1. Prepare your dataset in the following structure:
   ```
   data/
   ├── train/
   │   ├── Angry/
   │   ├── Disgusted/
   │   ├── Fearful/
   │   ├── Happy/
   │   ├── Neutral/
   │   ├── Sad/
   │   └── Surprised/
   ├── validation/
   │   └── (same structure as train)
   └── test/
       └── (same structure as train)
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

3. Evaluate the model performance:
   ```bash
   python test_model.py
   ```

## Engagement Metrics

The system calculates engagement based on four key metrics with the following weights:

- **Emotion Score (30%)**: Weighted values assigned to different emotions
- **Smile Detection (10%)**: Whether the user is smiling
- **Head Position (40%)**: Whether the head is stable or tilted
- **Gaze Direction (20%)**: Whether the user is looking directly at the camera

The final engagement level is classified as:
- **Fully Engaged**: Score > 59.9%
- **Partially Engaged**: Score between 49.9% and 59.9%
- **Not Engaged**: Score < 49.9%

## Project Structure

```
Real-Time-Emotion-Detection/
├── data/                  # Training data (not included in repo)
├── extension/             # Chrome extension files
│   ├── background.js      # Extension background service worker
│   ├── content.js         # Content script for video analysis
│   ├── manifest.json      # Extension manifest
│   ├── icons/             # Extension icons
│   └── popup/             # Extension popup interface
│       ├── popup.html     # Popup HTML interface
│       ├── popup.css      # Popup styles
│       └── popup.js       # Popup functionality
├── model/                 # Pre-trained emotion recognition model
├── emotion_score.py       # Emotion scoring utilities
├── main.py                # Main application for webcam analysis
├── server.py              # Flask API server
├── test_model.py          # Script to evaluate model performance
├── train_model.py         # Script to train the model
├── LICENSE.md             # MIT License
└── README.md              # Project documentation
```

## Chrome Extension Details

The "EngageAI" Chrome extension includes:

- **Background Service**: Handles extension state management and icon updates
- **Content Script**: Processes video frames and communicates with the Flask server
- **Popup Interface**: User-friendly controls and real-time metrics display
- **Permissions**:
  - `activeTab`: To access the current tab's content
  - `storage`: To save extension settings
  - `nativeMessaging`: For communication with the local server
  - `scripting`: For injecting content scripts

The extension specifically targets video conferencing platforms and provides real-time feedback through an overlay display during meetings.

## Facial Landmarks

The system uses MediaPipe's face mesh to detect 468 facial landmarks. Key landmark points used include:

- **Mouth**: Points 61, 291, 13, 14 for smile detection
- **Eyebrows**: Points 159, 145 for expression analysis
- **Eyes**: Points 33, 143, 159, 145, 362, 263, 386, 374 for gaze tracking
- **Face**: Points 4, 152 for head position analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- FER2013 dataset for emotion recognition training
- MediaPipe team for face mesh implementation
- TensorFlow team for deep learning framework
- Chrome Extensions API developers