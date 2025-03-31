import numpy as np
def get_emotion_score(emotion):
    emotion_scores = {
        "Angry" : 0.6, 
        "Disgusted" : 0.1, 
        "Fearful" : 0.4, 
        "Happy" : 1.0, 
        "Neutral" : 0.5, 
        "Sad" : 0.2, 
        "Surprised" : 0.8
    }
    return emotion_scores.get(emotion, 0.0)

def get_facial_features(landmarks):

    # facial features for mouth
    mouth_left = np.array([landmarks[61].x, landmarks[61].y])
    mouth_right = np.array([landmarks[291].x, landmarks[291].y])
    mouth_top = np.array([landmarks[13].x, landmarks[13].y])
    mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])

    # calculate smile factor
    mouth_width = np.linalg.norm(mouth_right - mouth_left)
    mouth_height = np.linalg.norm(mouth_top - mouth_bottom)
    smile_factor = mouth_width / mouth_height

    # facial features for eyebrows
    brow_left_top = np.array([landmarks[159].x, landmarks[159].y])
    brow_left_bottom = np.array([landmarks[145].x, landmarks[145].y])

    # calculate eyebrows raised
    eyebrows_distance = np.linalg.norm(brow_left_top - brow_left_bottom)

    # define the threshold
    smile_threshold = 1.8
    eyebrows_threshold = 0.05

    smile = smile_factor > smile_threshold
    eyebrow = eyebrows_distance < eyebrows_threshold

    return smile, eyebrow