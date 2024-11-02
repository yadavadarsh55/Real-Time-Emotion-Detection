import cv2
import numpy as np
from tensorflow.keras.models import load_model

emotion_model = load_model('model/emotion_recognition_model_final.h5')

haarcascade_face = "model/haarcascade_frontalface_default.xml"

emotion_list = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def preprocess_face(face_image):
    if face_image is None or face_image.size == 0:
        return None
    face_img = cv2.resize(face_image, (48, 48)) 
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1) 
    return face_img

def predict_emotion(face_image):
    # Preprocess the face image for emotion prediction
    processed_face = preprocess_face(face_image)
    if processed_face is None:
        return None
    emotion_prediction = emotion_model.predict(processed_face)
    emotion_label = np.argmax(emotion_prediction)  # Get the predicted emotion label
    return emotion_label

def Real_Time_video():

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
     
        success, img = cap.read()

        facecascade = cv2.CascadeClassifier(haarcascade_face)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face = facecascade.detectMultiScale(img_grey, 1.1, 4)

        for (x,y,w,h) in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            face_region = img_grey[x:x+w, y:y+h]

            emotion_label = predict_emotion(face_region)
            emotion_text = f'Emotion: {emotion_list[emotion_label]}'

            cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        cv2.imshow("Face", img)

        if cv2.waitKey(1) & 0xFF == ord('f'):
            break

Real_Time_video()
