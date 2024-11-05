import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

emotion_model = load_model('model/emotion_recognition_model_final.h5')

emotion_list = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

drawing_spec_connections = mp_drawing.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(255, 255, 255) 
)

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

def Real_Time_video():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame")
            break

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)

        if mesh_results.multi_face_landmarks:
            overlay = img.copy()
            for face_landmarks in mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(255,255,255), thickness=1, circle_radius=1)
                )
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

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

                if emotion_label is not None:
                    emotion_text = f'Emotion: {emotion_list[emotion_label]}'
                    cv2.putText(img, emotion_text, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face", img)

        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    face_detection.close()

Real_Time_video()
