main.py

import os
import shutil
import time
import pickle
import cv2
from image_processing import preprocess_image, process_video
from my_utils import species_mapping
from model import predict_species
from my_utils import (
    display_results, save_result, correct_prediction,
    save_corrected_data, save_filtered_result,
    preprocess_and_predict, parse_timestamp, group_images_by_time,
    process_image_group, handle_user_confirmation, ensure_directory_exists,
    adjust_confidence_for_time_group, load_reference_histograms, classify_image_based_on_histogram
)

# Load the elephant histograms from the reference file
with open('C:/Users/buddh/Desktop/VS Code/ANIMATE/src/reference_histograms.pkl', 'rb') as f:
    histograms = pickle.load(f)

# Elephant day and night histograms
elephant_histogram_day = histograms['elephant_day']
elephant_histogram_night = histograms['elephant_night']

# Function to compare histograms and return a match score
def compare_histograms(image_histogram, reference_histogram):
    score = cv2.compareHist(image_histogram, reference_histogram, cv2.HISTCMP_CORREL)
    return score

def calculate_histogram(image):
    """ Calculate the histogram for the given image, ensuring it has 3 channels """
    if image is None:
        raise ValueError("Image is None and cannot be processed.")
    if len(image.shape) == 2:
        # Image is grayscale, convert to 3-channel BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] != 3:
        raise ValueError("Image must have 3 channels (RGB or BGR) for histogram calculation.")
    
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()

def process_images_from_directory(directory, filter_species=None):
    # Load reference histograms
    reference_histograms = load_reference_histograms()

    # Group images by time (120-second window for adjusting confidence)
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png"))]
    grouped_images = group_images_by_time(image_paths, time_window=2)  # 120-second group

    for image_group in grouped_images:
        print(f"Processing image group: {image_group}")
        species_detected = process_image_group(image_group)

        for image_path in image_group:
            try:
                start_time = time.time()
                
                # Load and preprocess the image
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None or len(image.shape) != 3 or image.shape[2] != 3:
                    print(f"Error: Image at {image_path} has an unexpected shape: {image.shape if image is not None else 'None'}")
                    raise ValueError("Image must have 3 channels (RGB or BGR) for histogram calculation.")
                
                # Preprocess the image
                processed_image = preprocess_image(image_path)
                
                predictions = preprocess_and_predict(image_path)

                # Calculate histogram for the current image
                image_histogram = calculate_histogram(image)

                # Compare with elephant histograms (day and night)
                score_day = compare_histograms(image_histogram, elephant_histogram_day)
                score_night = compare_histograms(image_histogram, elephant_histogram_night)

                # Adjust confidence if the histogram matches the elephant histograms
                histogram_match_threshold = 0.8  # Adjust this threshold as needed
                if score_day >= histogram_match_threshold or score_night >= histogram_match_threshold:
                    print(f"Histogram match for elephant detected in {image_path}")
                    for i, prediction in enumerate(predictions):
                        if len(prediction) == 3:  # Check if the prediction has three elements
                            label, confidence = prediction[1], prediction[2]
                            if 'elephant' in label.lower() or 'tusker' in label.lower():
                                predictions[i] = (prediction[0], label, confidence * 1.5)  # Boost confidence by 50%
                        else:
                            print(f"Unexpected prediction format: {prediction}")
                            continue

                # Apply confidence adjustment BEFORE deciding to move to the Elephant folder
                adjusted_predictions = adjust_confidence_for_time_group(predictions, species_detected)
                top_prediction = adjusted_predictions[0]
                confidence = top_prediction[2]
                label = top_prediction[1].lower()

                display_results(adjusted_predictions)

                # Check if it's classified as "Elephant" or "Tusker"
                if 'elephant' in label or 'tusker' in label:
                    # Check confidence after adjustment
                    if confidence >= 0.30:  # Updated confidence threshold for elephants and tuskers
                        # Move image to Elephant folder
                        category_dir = save_result(image_path, adjusted_predictions)
                        if category_dir:
                            os.makedirs(category_dir, exist_ok=True)
                            destination_path = os.path.join(category_dir, os.path.basename(image_path))
                            print(f"Moving image to: {destination_path}")
                            if not os.path.exists(destination_path):
                                shutil.move(image_path, destination_path)
                        else:
                            print(f"Low confidence: {confidence}, moving image to Nondetect")
                            move_to_nondetect(image_path)
                    else:
                        # If confidence is too low, move to Nondetect
                        print(f"Low confidence: {confidence}, moving image to Nondetect")
                        move_to_nondetect(image_path)
                else:
                    # Automatically move all non-elephant images to Nondetect folder
                    print(f"Non-elephant detection: {label}, moving image to Nondetect")
                    move_to_nondetect(image_path)
            except ValueError as e:
                print(f"Error processing image {image_path}: {e}")

def move_to_nondetect(image_path):
    """ Helper function to move images to the Nondetect folder """
    category_dir = os.path.join('C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results', 'Nondetect')
    ensure_directory_exists(category_dir)
    destination_path = os.path.join(category_dir, os.path.basename(image_path))
    print(f"Moving image to Nondetect: {destination_path}")
    if not os.path.exists(destination_path):
        shutil.move(image_path, destination_path)

def process_videos_from_directory(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith((".mp4", ".avi")):  # Make extension check case-insensitive
            video_path = os.path.join(directory, filename)
            print(f"Processing {video_path}")
            process_video(video_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze images or videos.")
    parser.add_argument("--type", type=str, choices=["images", "videos"], required=True, help="Type of files to analyze: 'images' or 'videos'")
    parser.add_argument("--retrain", action='store_true', help="Retrain the model with corrected data")
    parser.add_argument("--fetch", action='store_true', help="Fetch data from GBIF")
    parser.add_argument("--filter", type=str, help="Filter species, e.g., 'elephant'")
    parser.add_argument("--local-only", action='store_true', help="Run program without using Azure API (local predictions only)")
    args = parser.parse_args()

    if args.fetch:
        # Remove GBIF related code
        pass

    if args.type:
        if args.type == "images":
            image_directory = "data/images"
            process_images_from_directory(image_directory, args.filter)
        elif args.type == "videos":
            video_directory = "data/videos"
            process_videos_from_directory(video_directory)

    if args.retrain:
        retrain_model()
