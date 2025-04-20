import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import statistics
import base64
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load emotion recognition model
emotion_model = load_model('model/emotion_recognition_model_final.h5')

# Define emotion list
emotion_list = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# MediaPipe configurations
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize face detection and mesh
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5,
    model_selection=1
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class EngagementAnalyzer:
    def __init__(self, time_span=10):
        self.time_span = time_span
        self.start_time = None
        self.emotion_scores = []
        self.smile_scores = []
        self.head_scores = []
        self.gaze_scores = []
        self.last_engagement_score = 0.0
        self.last_engagement_level = "Not Engaged"

    def reset_metrics(self):
        self.emotion_scores = []
        self.smile_scores = []
        self.head_scores = []
        self.gaze_scores = []
        self.start_time = time.time()

    def add_frame_metrics(self, emotion_score, smile_score, head_score, gaze_score):
        if self.start_time is None:
            self.start_time = time.time()
        
        self.emotion_scores.append(emotion_score)
        self.smile_scores.append(smile_score)
        self.head_scores.append(head_score)
        self.gaze_scores.append(gaze_score)

    def calculate_time_span_engagement(self):
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        if current_time - self.start_time < self.time_span:
            return self.last_engagement_score, self.last_engagement_level

        try:
            avg_emotion_score = statistics.mean([s for s in self.emotion_scores if s is not None]) if self.emotion_scores else 0
            avg_smile_score = statistics.mean([s for s in self.smile_scores if s is not None]) if self.smile_scores else 0
            avg_head_score = statistics.mean([s for s in self.head_scores if s is not None]) if self.head_scores else 0
            avg_gaze_score = statistics.mean([s for s in self.gaze_scores if s is not None]) if self.gaze_scores else 0

            weight_smile = 0.1
            weight_head = 0.4
            weight_emotion = 0.3
            weight_gaze = 0.2

            engagement_score = (
                (weight_smile * avg_smile_score) + 
                (weight_head * avg_head_score) + 
                (weight_emotion * avg_emotion_score) +
                (weight_gaze * avg_gaze_score)
            )
            
            if engagement_score * 100 > 59.9:
                engagement_level = "Fully Engaged"
            elif 49.9 < engagement_score * 100 <= 59.9:
                engagement_level = "Partially Engaged"
            else:
                engagement_level = "Not Engaged"

            self.last_engagement_score = engagement_score
            self.last_engagement_level = engagement_level
            self.reset_metrics()

            return self.last_engagement_score, self.last_engagement_level
        except Exception as e:
            self.last_engagement_score = 0
            print(f"Error calculating engagement: {e}")
            return 0.0, "Not Engaged"

# Initialize engagement analyzer
engagement_analyzer = EngagementAnalyzer(time_span=3)

def preprocess_face(face_image):
    if face_image is None or face_image.size == 0:
        return None
    face_img = cv2.resize(face_image, (48, 48)) 
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1) 
    return face_img

def predict_emotion(face_image):
    processed_face = preprocess_face(face_image)
    if processed_face is None:
        return None
    emotion_prediction = emotion_model.predict(processed_face)
    emotion_label = np.argmax(emotion_prediction)  
    return emotion_label

def predict_emotion_score(emotion):
    emotion_scores = {
        "Angry": 0.6, 
        "Disgusted": 0.1, 
        "Fearful": 0.4, 
        "Happy": 1.0, 
        "Neutral": 0.8, 
        "Sad": 0.2, 
        "Surprised": 0.9
    }
    return emotion_scores.get(emotion, 0.0)

def detect_smile(landmarks, frame_width, frame_height):
    mouth_left = np.array([landmarks[61].x * frame_width, landmarks[61].y * frame_height])
    mouth_right = np.array([landmarks[291].x * frame_width, landmarks[291].y * frame_height])
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    jaw_left = np.array([landmarks[147].x * frame_width, landmarks[147].y * frame_height])
    jaw_right = np.array([landmarks[376].x * frame_width, landmarks[376].y * frame_height])
    jaw_width = np.linalg.norm(jaw_right - jaw_left)
    smile_factor = mouth_width / jaw_width

    smile_threshold = 0.485
    smile = smile_factor > smile_threshold
    return 1.0 if smile else 0

def detect_head_movements(landmarks, frame_width, frame_height):
    nose = np.array([landmarks[4].x * frame_width, landmarks[4].y * frame_height])
    chin = np.array([landmarks[152].x * frame_width, landmarks[152].y * frame_height])
    tilt_angle = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))

    if tilt_angle > 105:
        return 0.6
    if tilt_angle < 80:
        return 0.6
    return 1.0

def detect_gaze(landmarks, frame_width, frame_height):
    right_eye = [landmarks[33], landmarks[143], landmarks[159], landmarks[145]]
    right_eye_points = [(int(pt.x * frame_width), int(pt.y * frame_height)) for pt in right_eye]
    left_eye = [landmarks[362], landmarks[263], landmarks[386], landmarks[374]]
    left_eye_points = [(int(pt.x * frame_width), int(pt.y * frame_height)) for pt in left_eye]

    def calc_eye_gaze(eye_points):
        outer, inner, top, bottom = eye_points
        eye_width = abs(outer[0] - inner[0])
        eye_height = abs(top[1] - bottom[1])
        iris_x = (inner[0] + outer[0])//2
        iris_y = (top[1] + bottom[1])//2

        if eye_width != 0 and eye_height != 0:
            ratio_x = abs(iris_x - outer[0]) / eye_width
            ratio_y = abs(iris_y - top[1]) / eye_height
            return ratio_x, ratio_y
        return None
    
    right_gaze = calc_eye_gaze(right_eye_points)
    left_gaze = calc_eye_gaze(left_eye_points)
    if right_gaze and left_gaze:
        avg_x_ratio = (right_gaze[0] + left_gaze[0]) / 2
        avg_y_ratio = (right_gaze[1] + left_gaze[1]) / 2
        if (0.499 < avg_x_ratio < 0.510) or (0.499 < avg_y_ratio < 0.510):
            return 1.0
    return 0

def base64_to_image(base64_string):
    try:
        base64_data = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_frame():
    try:
        data = request.json
        if not data or 'frame' not in data:
            return jsonify({"error": "No frame data provided"}), 400

        frame = base64_to_image(data['frame'])
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        if not detection_results.detections:
            return jsonify({
                "status": "no_face",
                "message": "No face detected"
            })

        response_data = {"faces": []}

        for detection in detection_results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            face_region = img_grey[y:y+height, x:x+width]
            emotion_label = predict_emotion(face_region)
            
            if emotion_label is None:
                continue

            emotion = emotion_list[emotion_label]
            emotion_score = predict_emotion_score(emotion)
            smile_score = 0
            head_score = 0
            gaze_score = 0

            if mesh_results.multi_face_landmarks:
                for landmarks in mesh_results.multi_face_landmarks:
                    smile_score = detect_smile(landmarks.landmark, w, h)
                    head_score = detect_head_movements(landmarks.landmark, w, h)
                    gaze_score = detect_gaze(landmarks.landmark, w, h)

                    engagement_analyzer.add_frame_metrics(
                        emotion_score, 
                        smile_score, 
                        head_score, 
                        gaze_score
                    )

            engagement_score, engagement_level = engagement_analyzer.calculate_time_span_engagement()

            response_data["faces"].append({
                "emotion": emotion,
                "emotion_score": float(emotion_score),
                "smile_score": float(smile_score),
                "head_score": float(head_score),
                "gaze_score": float(gaze_score),
                "engagement_score": float(engagement_score),
                "engagement_level": engagement_level,
                "bounding_box": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            })

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "running",
        "model_loaded": emotion_model is not None,
        "analyzer_ready": True
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)