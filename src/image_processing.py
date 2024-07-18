import cv2
import numpy as np
from model import predict_species
from my_utils import display_results, save_frame_result
from helpers import preprocess_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = preprocess_frame(frame)
        predictions = predict_species(processed_frame)
        display_results(predictions)
        save_frame_result(frame, predictions, frame_count)
        frame_count += 1
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def preprocess_frame(frame):
    img_array = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
