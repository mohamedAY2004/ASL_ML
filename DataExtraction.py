"""
Created on Sun May  4 21:58:20 2025

@author: Mohamed
"""
import os
import cv2
import pandas as pd
import mediapipe as mp


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=1)

# Set your dataset directory
dataset_path = 'C:\\Users\\Mohamed\\Downloads\\ASL_Alphabet_Dataset\\asl_alphabet_train'

# Final DataFrame
data = []

# Iterate over all folders
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    print(folder_path)
    if not os.path.isdir(folder_path):
        continue  # Skip files

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue  # Skip non-image files

        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip unreadable images
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])  # Flatten x, y, z
            if len(coords) == 63:  # 21 landmarks * 3
                coords.append(folder_name)  # Add label
                data.append(coords)

# Create DataFrame
columns = [f'{axis}{i+1}' for i in range(21) for axis in ['x', 'y', 'z']] + ['letter']
df = pd.DataFrame(data, columns=columns)
# Drop duplicate rows
df = df.drop_duplicates()
# (Optional) Save the cleaned data to a new CSV file
df.to_csv('cleaned_file.csv', index=False)