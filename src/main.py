import os
import shutil
from image_processing import preprocess_image, process_video
from model import predict_species
from my_utils import (
    display_results, save_result, correct_prediction,
    save_corrected_data, save_filtered_result,
    preprocess_and_predict, parse_timestamp, group_images_by_time,
    process_image_group, handle_user_confirmation, ensure_directory_exists,
    adjust_confidence_based_on_group
)

def process_images_from_directory(directory, filter_species=None):
    # Group images by time
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png"))]
    grouped_images = group_images_by_time(image_paths)

    for image_group in grouped_images:
        print(f"Processing image group: {image_group}")
        species_detected = process_image_group(image_group)
        
        for image_path in image_group:
            processed_image = preprocess_image(image_path)
            predictions = preprocess_and_predict(image_path)
            display_results(predictions)
            
            # Adjust confidence based on the group
            adjusted_predictions = adjust_confidence_based_on_group(predictions, species_detected)
            top_prediction = adjusted_predictions[0]
            confidence = top_prediction[2]
            
            if confidence >= 0.34:
                category_dir = save_result(image_path, adjusted_predictions)
                if category_dir:
                    os.makedirs(category_dir, exist_ok=True)
                    destination_path = os.path.join(category_dir, os.path.basename(image_path))
                    print(f"Moving image to: {destination_path}")
                    if not os.path.exists(destination_path):
                        shutil.move(image_path, destination_path)
            else:
                correct_label = handle_user_confirmation(image_path, adjusted_predictions)
                print(f"User corrected label: {correct_label}")
                correct_prediction(image_path, correct_label)
                save_corrected_data(image_path, correct_label)
                # Explicitly handle "Elephant" correction
                if 'elephant' in correct_label.lower() or 'tusker' in correct_label.lower():
                    corrected_dir = os.path.join('C:/Users/buddh/Desktop/VS Code/ANIMATE/data/results', 'Elephant')
                    ensure_directory_exists(corrected_dir)
                    destination_path = os.path.join(corrected_dir, os.path.basename(image_path))
                    print(f"Explicitly moving corrected Elephant image to: {destination_path}")
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
