import json
import pickle
import cv2
import numpy as np
#import pandas as pd
#import librosa
import tensorflow as tf

from flask import Flask, request, jsonify, url_for, render_template

app = Flask(__name__)

# Load video model
file_path = 'E:/sightica/ML endtoend/realtime_fall/clf.pkl'
with open(file_path, 'rb') as f:
    clf = pickle.load(f)

# Load audio model
file_path = 'E:/sightica/ML endtoend/realtime_fall/gradient_booster.pkl'
with open(file_path, 'rb') as f:
    gradient_booster = pickle.load(f)

def calculate_centroid(keypoints):
    x_coords = keypoints[:, 0]
    y_coords = keypoints[:, 1]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return np.array([centroid_x, centroid_y])

# Function to calculate angles between keypoints
def calculate_angles(keypoints):
    # Calculate angles between hip, knee, and ankle keypoints for each leg
    left_hip, left_knee, left_ankle = keypoints[11], keypoints[12], keypoints[13]
    right_hip, right_knee, right_ankle = keypoints[8], keypoints[9], keypoints[10]
    left_leg_angle = np.arctan2(left_ankle[1] - left_knee[1], left_ankle[0] - left_knee[0]) \
                      - np.arctan2(left_hip[1] - left_knee[1], left_hip[0] - left_knee[0])
    right_leg_angle = np.arctan2(right_ankle[1] - right_knee[1], right_ankle[0] - right_knee[0]) \
                       - np.arctan2(right_hip[1] - right_knee[1], right_hip[0] - right_knee[0])
    # Calculate angles between shoulder, elbow, and wrist keypoints for each arm
    left_shoulder, left_elbow, left_wrist = keypoints[5], keypoints[6], keypoints[7]
    right_shoulder, right_elbow, right_wrist = keypoints[2], keypoints[3], keypoints[4]
    left_arm_angle = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0]) \
                      - np.arctan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0])
    right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0]) \
                       - np.arctan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0])
    return np.array([left_leg_angle, right_leg_angle, left_arm_angle, right_arm_angle])

# Function to check if feet keypoints are in contact with the ground
def check_feet_contact(keypoints):
    left_foot, right_foot = keypoints[14], keypoints[11]
    if left_foot[1] > right_foot[1]:
        lower_foot = right_foot
        higher_foot = left_foot
    else:
        lower_foot = left_foot
        higher_foot = right_foot
    return lower_foot[1] <= higher_foot[1]

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize audio recording
rate = 16000
chunk = int(rate/10)
audio_data = np.zeros(chunk)

# TensorFlow Lite interpreter for pose estimation
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Define route for fall detection
@app.route('/fall_detection', methods=['POST'])
def fall_detection():
    # Capture video frame
    ret, frame = cap.read()

    # Extract video features from frame
    img = cv2.resize(frame, (192, 192))
    input_image = tf.cast(img, dtype=tf.float32)

    # Add batch size dimension to input image
    input_image = input_image[tf.newaxis, ...]
  

    # Setup input and output for TensorFlow Lite interpreter
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions with video model
    interpreter.set_tensor(input_details[0]['index'], input_image)
    #interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints_with_scores.reshape([17, 3])
    keypoints_data = keypoints_with_scores.reshape([17, 3])[:, :2]
    centroid = calculate_centroid(keypoints_data)
    keypoints_data = np.array(keypoints_data)
    keypoints_data = np.squeeze(keypoints_data)
    angles = calculate_angles(keypoints_data)
    feet_contact = check_feet_contact(keypoints_data)
    features = np.concatenate((centroid, angles, [feet_contact]))
    feature_vector = np.concatenate((keypoints_data.flatten(), features), axis=0)
    feature = feature_vector.reshape(1, -1)
    video_pred = clf.predict(feature)

    # Record audio data
    # data = stream.read(chunk)
    # data_np = np.frombuffer(data, dtype=np.float32)
    # audio_data = np.concatenate((audio_data, data_np), axis=0)

    # # Extract audio features from audio data
    # audio_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=rate, n_mfcc=50).T, axis=0)
    # audio_pred = gradient_booster.predict(audio_features.reshape(1, -1))

    # Implement decision logic
    if video_pred == 1:
        result = "Fall detected!"
    else:
        result = "No fall"

    return jsonify(result=result)

# Define home route for displaying video frame
@app.route('/')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
