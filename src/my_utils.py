import os
import shutil
import requests
from datetime import datetime, timedelta
from PIL import Image
from PIL.ExifTags import TAGS
from helpers import preprocess_image
from model import predict_species
from azure_config import AZURE_ENDPOINT, AZURE_KEY  # Import Azure configuration

# Mapping from model species names to provided species list
species_mapping = {
    'elephant': 'Elephant',
    'tusker': 'Elephant',
    'gorilla': 'Gorilla',
    'chimpanzee': 'Chimpanzee',
    'mandrill': 'Mandrill',
    'guenon': 'Guenon',
    'mangabey': 'Mangabey',
    'talapoin': 'Talapoin',
    'sitatunga': 'Sitatunga',
    'buffalo': 'Buffalo',
    'duiker': 'Duiker',
    'pangolin': 'Pangolin',
    'mongoose': 'Mongoose',
    'hog': 'Hog',
    'rat': 'Rat',
    'genet': 'Genet',
    'bat': 'Bat',
    'guineafowl': 'Guineafowl',
    'other': 'Other'
}

def azure_analyze_image(image_path):
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_KEY,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'visualFeatures': 'Tags,Description',
        'details': 'Landmarks'
    }
    with open(image_path, 'rb') as image_data:
        response = requests.post(AZURE_ENDPOINT + "/vision/v3.2/analyze", headers=headers, params=params, data=image_data)
    response.raise_for_status()
    analysis = response.json()
    return analysis

def display_results(predictions):
    for prediction_set in predictions:
        if isinstance(prediction_set, list):
            for prediction in prediction_set:
                if len(prediction) == 3:
                    _, name, score = prediction
                    species = map_species(name)
                    print(f"{species}: {score:.2f}")
                else:
                    print(f"Unexpected prediction format: {prediction}")
        else:
            if len(prediction_set) == 3:
                _, name, score = prediction_set
                species = map_species(name)
                print(f"{species}: {score:.2f}")
            else:
                print(f"Unexpected prediction format: {prediction_set}")

def map_species(name):
    name = name.lower()
    if 'elephant' in name or 'tusker' in name:
        return 'Elephant'
    for key in species_mapping:
        if key in name:
            return species_mapping[key]
    return name.capitalize()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_result(image_path, predictions, base_dir='C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results'):
    top_prediction = predictions[0]
    name = top_prediction[1]
    species = map_species(name)
    confidence = top_prediction[2]

    if confidence >= 0.34:
        if 'elephant' in species.lower() or 'tusker' in species.lower():
            category_dir = os.path.join(base_dir, 'Elephant')
        else:
            category_dir = os.path.join(base_dir, 'Other', get_species_category(species), species)
    else:
        return None  # Return None if confidence is low
    
    ensure_directory_exists(category_dir)
    return category_dir

def get_species_category(species):
    primates = ['Gorilla', 'Chimpanzee', 'Mandrill', 'Guenon', 'Mangabey', 'Talapoin', 'Other']
    ungulates = ['Sitatunga', 'Buffalo', 'Duiker']
    others = ['Pangolin', 'Mongoose', 'Hog', 'Rat', 'Genet', 'Bat', 'Guineafowl', 'Other']

    if species in primates:
        return 'Primates'
    elif species in ungulates:
        return 'Ungulates'
    elif species in others:
        return 'Other'
    else:
        return 'Other'

def save_filtered_result(image_path, predictions, filter_species, base_dir='C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results'):
    top_prediction = predictions[0]
    name = top_prediction[1]
    if filter_species.lower() in name.lower():
        category_dir = os.path.join(base_dir, 'Elephant')
    else:
        category_dir = os.path.join(base_dir, 'Other')
    
    ensure_directory_exists(category_dir)
    
    if category_dir == os.path.join(base_dir, 'Other'):
        other_species = get_general_species(name)
        other_species_dir = os.path.join(category_dir, other_species)
        ensure_directory_exists(other_species_dir)
        category_dir = other_species_dir

    destination_path = os.path.join(category_dir, os.path.basename(image_path))
    if not os.path.exists(destination_path):
        shutil.move(image_path, destination_path)

def get_species_from_azure(tags):
    for tag in tags:
        if tag['name'].lower() in species_mapping:
            return species_mapping[tag['name'].lower()]
    return None

def preprocess_and_predict(image_path):
    processed_image = preprocess_image(image_path)
    predictions = predict_species(processed_image)
    azure_analysis = azure_analyze_image(image_path)
    azure_species = get_species_from_azure(azure_analysis['tags'])

    if azure_species:
        predictions.append((None, azure_species, 0.5))  # Assign a default confidence score for Azure tags

    return predictions

def save_frame_result(frame, predictions, frame_count, base_dir='C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results'):
    pass  # Functionality moved to correct_prediction to handle after user confirmation

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

def correct_prediction(image_path, correct_label, base_dir='C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results'):
    if 'elephant' in correct_label.lower() or 'tusker' in correct_label.lower():
        corrected_dir = os.path.join(base_dir, 'Elephant')
        print(f"Correcting Elephant to directory: {corrected_dir}")
    else:
        category = get_species_category(map_species(correct_label))
        corrected_dir = os.path.join(base_dir, 'Other', category, map_species(correct_label))
        print(f"Correcting {correct_label} to directory: {corrected_dir}")
    
    ensure_directory_exists(corrected_dir)
    
    # Move the image to the correct folder
    destination_path = os.path.join(corrected_dir, os.path.basename(image_path))
    print(f"Moving corrected image to: {destination_path}")
    if not os.path.exists(destination_path):
        # Remove the image from any existing location in the results directory
        for root, _, files in os.walk(base_dir):
            if os.path.basename(image_path) in files:
                os.remove(os.path.join(root, os.path.basename(image_path)))
        shutil.move(image_path, destination_path)
    else:
        # If the image already exists in the destination, remove the original to avoid duplication
        os.remove(image_path)

def save_corrected_data(image_path, correct_label):
    corrected_data_path = 'data/corrected_data.txt'
    with open(corrected_data_path, 'a') as f:
        f.write(f"{image_path},{correct_label}\n")

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

def handle_user_confirmation(image_path, predictions):
    confirmed = False
    while not confirmed:
        correct = input(f"Is the prediction correct for {image_path}? (y/n): ")
        if correct.lower() in ['y', 'n']:
            confirmed = True
            if correct.lower() == 'n':
                correct_label = input(f"Enter the correct label for {image_path}: ")
                if 'elephant' in correct_label.lower() or 'elephants' in correct_label.lower() or 'tusker' in correct_label.lower():
                    print(f"User confirmed Elephant for {image_path}")
                    return 'Elephant'
                return correct_label
            else:
                top_prediction = predictions[0]
                if 'elephant' in top_prediction[1].lower() or 'elephants' in top_prediction[1].lower() or 'tusker' in top_prediction[1].lower():
                    print(f"Top prediction confirmed as Elephant for {image_path}")
                    return 'Elephant'
                return top_prediction[1]

def adjust_confidence_based_on_group(predictions, species_detected):
    adjusted_predictions = []
    for prediction in predictions:
        label = prediction[1]
        confidence = prediction[2]
        if label in species_detected and confidence < 0.34:
            confidence *= 2  # Adjust confidence, for example, double it
        adjusted_predictions.append((prediction[0], label, confidence))
    adjusted_predictions.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
    return adjusted_predictions
