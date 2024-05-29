import os
import shutil
import cv2
from datetime import datetime, timedelta
from PIL import Image
from PIL.ExifTags import TAGS
from helpers import preprocess_image
from model import predict_species

def display_results(predictions):
    for _, name, score in predictions:
        print(f"{name}: {score:.2f}")

def save_result(image_path, predictions, base_dir='data/results'):
    top_prediction = predictions[0]
    name = top_prediction[1]
    species = get_general_species(name)
    category_dir = os.path.join(base_dir, species)
    os.makedirs(category_dir, exist_ok=True)
    destination_path = os.path.join(category_dir, os.path.basename(image_path))
    if not os.path.exists(destination_path):
        shutil.copy(image_path, destination_path)

def save_frame_result(frame, predictions, frame_count, base_dir='data/results'):
    top_prediction = predictions[0]
    name = top_prediction[1]
    species = get_general_species(name)
    category_dir = os.path.join(base_dir, species)
    os.makedirs(category_dir, exist_ok=True)
    frame_path = os.path.join(category_dir, f"frame_{frame_count}.jpg")
    if not os.path.exists(frame_path):
        cv2.imwrite(frame_path, frame)

def get_general_species(name):
    if 'elephant' in name:
        return 'Elephant'
    elif 'gorilla' in name or 'siamang' in name or 'howler_monkey' in name:
        return 'Gorilla'
    elif 'butterfly' in name or 'monarch' in name or 'admiral' in name or 'lycaenid' in name:
        return 'Butterfly'
    else:
        return name.split('_')[0].capitalize()

def parse_timestamp(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()
    if not exif_data:
        return None
    
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        if decoded == 'DateTime':
            return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    return None

def group_images_by_time(image_paths, time_window=5):
    time_window = timedelta(minutes=time_window)
    groups = []
    current_group = []
    last_time = None

    for img_path in sorted(image_paths, key=lambda x: parse_timestamp(x) or datetime.min):
        timestamp = parse_timestamp(img_path)
        if timestamp is None:
            continue

        if last_time is None or (timestamp - last_time) <= time_window:
            current_group.append(img_path)
        else:
            groups.append(current_group)
            current_group = [img_path]
        last_time = timestamp

    if current_group:
        groups.append(current_group)

    return groups

def process_image_group(image_group):
    species_detected = {}
    for img_path in image_group:
        processed_image = preprocess_image(img_path)
        predictions = predict_species(processed_image)
        print(f"Processing {img_path}")
        for _, name, score in predictions:
            if name not in species_detected:
                species_detected[name] = score
            else:
                species_detected[name] = max(species_detected[name], score)

    # Sort species by confidence score and take top 3
    sorted_species = sorted(species_detected.items(), key=lambda item: item[1], reverse=True)[:3]

    # Print sorted species with scores
    for name, score in sorted_species:
        print(f"{name}: {score:.2f}")

    return [name for name, _ in sorted_species]
