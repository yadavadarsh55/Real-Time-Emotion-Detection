import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import statistics

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

# Drawing specifications
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

drawing_spec_connections = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(255, 255, 255) 
)

class EngagementAnalyzer:
    def __init__(self, time_span=10):
        self.time_span = time_span
        self.start_time = None
        
        # Lists to store metrics over the time span
        self.emotion_scores = []
        self.smile_scores = []
        self.head_scores = []
        self.gaze_scores = []

        self.last_engagement_score = 0.0
        self.last_engagement_level = "Not Engaged"
        self.display_duration = time_span

    def reset_metrics(self):
        """Reset all metric collections"""
        self.emotion_scores = []
        self.smile_scores = []
        self.head_scores = []
        self.gaze_scores = []
        self.start_time = time.time()

    def add_frame_metrics(self, emotion_score, smile_score, head_score, gaze_score):
        """
        Add frame metrics to the collection
        """
        # If this is the first frame, start the timer
        if self.start_time is None:
            self.start_time = time.time()
        
        # Store frame metrics
        self.emotion_scores.append(emotion_score)
        self.smile_scores.append(smile_score)
        self.head_scores.append(head_score)
        self.gaze_scores.append(gaze_score)

    def calculate_time_span_engagement(self):
        """
        Calculate engagement score for the time span
        """
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        
        # Check if time span has elapsed
        current_time = time.time()
        if current_time - self.start_time < self.time_span:
            return self.last_engagement_score, self.last_engagement_level

        # Calculate average scores
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

            # Determine engagement level
            if engagement_score * 100 > 59.9:
                engagement_level = "Fully Engaged"
            elif 49.9 < engagement_score * 100 <= 59.9:
                engagement_level = "Partially Engaged"
            else:
                engagement_level = "Not Engaged"

            # Store last computed values
            self.last_engagement_score = engagement_score
            self.last_engagement_level = engagement_level

            # Reset metrics for the next time span
            self.reset_metrics()

            return self.last_engagement_score, self.last_engagement_level
        except Exception as e:
            self.last_engagement_score = 0
            print(f"Error calculating engagement: {e}")
            return 0.0, "Not Engaged"

# Existing functions from original code
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
        "Angry" : 0.6, 
        "Disgusted" : 0.1, 
        "Fearful" : 0.4, 
        "Happy" : 1.0, 
        "Neutral" : 0.8, 
        "Sad" : 0.2, 
        "Surprised" : 0.9
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

    if(smile):
        return 1.0
    else:
        return 0

def detect_head_movements(landmarks, frame_width, frame_height):
    nose = np.array([landmarks[4].x * frame_width, landmarks[4].y * frame_height])
    chin = np.array([landmarks[152].x * frame_width, landmarks[152].y * frame_height])

    head_text = "Stable"
    tilt_angle = np.degrees(np.arctan2(chin[1] - nose[1], chin[0] - nose[0]))

    if tilt_angle > 105:
        head_text = "Tilted Left"
    if tilt_angle < 80:
        head_text = "Tilted Right"

    movement_score_list = {
        'Stable' : 1.0,                     
        'Tilted Left' : 0.6, 
        "Tilted Right" : 0.6
    }

    score = movement_score_list.get(head_text)

    return score

def detect_gaze(landmarks, frame_widht, frame_height):
    right_eye = [landmarks[33], landmarks[143], landmarks[159], landmarks[145]]
    right_eye_points = [(int(pt.x * frame_widht), int(pt.y * frame_height)) for pt in right_eye]

    left_eye = [landmarks[362], landmarks[263], landmarks[386], landmarks[374]]
    left_eye_points = [(int(pt.x * frame_widht), int(pt.y * frame_height)) for pt in left_eye]

    def calc_eye_gaze(eye_points):
        outer, inner, top, bottom = eye_points
        eye_width = abs(outer[0] - inner[0])
        eye_height = abs(top[1] - bottom[1])

        iris_x = (inner[0] + outer[0])//2
        iris_y = (top[1] + bottom[1])//2

        if(eye_width != 0 and eye_height != 0):
            ratio_x = abs(iris_x - outer[0]) / eye_width
            ratio_y = abs(iris_y - top[1]) / eye_height
            return ratio_x, ratio_y
        return None
    
    right_gaze = calc_eye_gaze(right_eye_points)
    left_gaze = calc_eye_gaze(left_eye_points)
    if right_gaze and left_gaze is not None:
        avg_x_ratio = (right_gaze[0] + left_gaze[0]) / 2
        avg_y_ratio = (right_gaze[1] + left_gaze[1]) / 2
        if((avg_x_ratio > 0.499 and avg_x_ratio < 0.510) or (avg_y_ratio > 0.499 and avg_y_ratio < 0.510)):
            return 1.0
        else:
            return 0

def calculate_engagement_score(emotion_score, smile_score, head_score, gaze_score):
    weight_smile = 0.1
    weight_head = 0.4
    weight_emotion = 0.3
    weight_gaze = 0.2
    try:
        engagement_score = (
            (weight_smile * smile_score) + 
            (weight_head * head_score) + 
            (weight_emotion * emotion_score) +
            (weight_gaze * gaze_score)
        )
        if(engagement_score*100 > 59.9 ):
            return engagement_score*100, "Fully Engaged"
        elif(engagement_score*100 <= 59.9 and engagement_score*100 > 49.9):
            return engagement_score*100, "Partially Engaged"
        else:
            return engagement_score*100, "Not Engaged"
    except:
        return 0.0, "Not Engaged"

def Real_Time_video():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Initialize the engagement analyzer
    engagement_analyzer = EngagementAnalyzer(time_span=3)  # 10-second time span

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame")
            break

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)

                cv2.rectangle(img, (x, y), (x + width, y + height), (255, 255, 255), 1)

                face_region = img_grey[y:y+height, x:x+width]
                emotion_label = predict_emotion(face_region)
                emotion_score = predict_emotion_score(emotion_list[emotion_label])

                if emotion_label is not None:
                    emotion_text = f'Emotion: {emotion_list[emotion_label]}'
                    score_text = f'Score: {emotion_score}'
                    cv2.putText(img, emotion_text, (x, y - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(img, score_text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if mesh_results.multi_face_landmarks:
                    for landmarks in mesh_results.multi_face_landmarks:
                        smile_score = detect_smile(landmarks.landmark, x + width, y + height)
                        head_score = detect_head_movements(landmarks.landmark, x + width, y + height)
                        gaze_score = detect_gaze(landmarks.landmark, x + width, y + height)
                        
                        # Add frame metrics to engagement analyzer
                        engagement_analyzer.add_frame_metrics(
                            emotion_score, 
                            smile_score, 
                            head_score, 
                            gaze_score
                        )

                        # Calculate engagement score every 10 seconds
                    engagement_result = engagement_analyzer.calculate_time_span_engagement()
                    if engagement_result:
                        engagement_score, engagement_text = engagement_result
                        engagement_score_text = f"10s Engagement: {round(engagement_score, 2)}%"
                        engagement_final_text = f"Status: {engagement_text}"
                        
                        # Display the time-span engagement results
                        cv2.putText(img, engagement_score_text, (20, 55), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, engagement_final_text, (20, 75), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Reset for next time span
                        # engagement_analyzer.reset_metrics()

        cv2.imshow("Face", img)

        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

    face_mesh.close()
    face_detection.close()
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time video function
Real_Time_video()