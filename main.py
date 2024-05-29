import os
from helpers import preprocess_image
from image_processing import process_video
from model import predict_species
from utils import display_results, save_result, group_images_by_time, process_image_group, parse_timestamp

def process_images_from_directory(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    grouped_images = group_images_by_time(image_paths)

    for group in grouped_images:
        if group:
            time_range = f"Time: {parse_timestamp(group[0])} - {parse_timestamp(group[-1])}" if parse_timestamp(group[0]) else "Unknown Time"
            print(time_range)
            species = process_image_group(group)
            print(f"Species Detected: {', '.join(species)}")
            print()

    # Process images without metadata individually
    for img_path in image_paths:
        if parse_timestamp(img_path) is None:
            print(f"Processing {img_path} (No metadata)")
            processed_image = preprocess_image(img_path)
            predictions = predict_species(processed_image)
            display_results(predictions)
            save_result(img_path, predictions)

def process_videos_from_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            video_path = os.path.join(directory, filename)
            print(f"Processing {video_path}")
            process_video(video_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze images or videos.")
    parser.add_argument("--type", type=str, choices=["images", "videos"], required=True, help="Type of files to analyze: 'images' or 'videos'")
    args = parser.parse_args()

    if args.type == "images":
        image_directory = "data/images"
        process_images_from_directory(image_directory)
    elif args.type == "videos":
        video_directory = "data/videos"
        process_videos_from_directory(video_directory)
