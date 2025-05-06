import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load saved model and label encoder
model = joblib.load('asl_random_forest_model.joblib')
label_encoder = joblib.load('asl_label_encoder.joblib')

# Function to normalize landmarks (must match training)
def normalize_landmarks_3d(landmarks):
    x_wrist, y_wrist, z_wrist = landmarks[0]
    x_tip, y_tip, z_tip = landmarks[8]

    scale = np.sqrt((x_tip - x_wrist)**2 + (y_tip - y_wrist)**2 + (z_tip - z_wrist)**2)
    if scale == 0:
        return np.zeros(63)

    normalized = []
    for x, y, z in landmarks:
        normalized.append((x - x_wrist) / scale)
        normalized.append((y - y_wrist) / scale)
        normalized.append((z - z_wrist) / scale)
    return np.array(normalized)

# Function to predict ASL letter from an image
def predict_letter_from_image(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

        # Normalize and reshape input
        normalized = normalize_landmarks_3d(landmarks).reshape(1, -1)
        print(normalized)
        prediction = model.predict(normalized)
        prediction = model.predict(normalized)[0]
        predicted_letter = label_encoder.inverse_transform([prediction])[0]
        return predicted_letter
    else:
        return "No hand detected."

# Example usage
ans=predict_letter_from_image('L_test.jpg')
