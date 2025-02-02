import os
import shutil
import time
import pickle
import cv2
import tensorflow as tf
from image_processing import preprocess_image, process_video
from model import predict_species, retrain_model
from my_utils import (
    display_results, ensure_directory_exists, preprocess_and_predict
)

# File paths
CAPUCHIN_MODEL_PATH = "C:/Users/buddh/Desktop/VS Code/ANIMATE/src/capuchin_model.h5"
CAPUCHIN_SOURCE_DIR = "D:/MIKAELA/Photos and videos/Camera traps brazil"
CAPUCHIN_TARGET_DIR = "D:/MIKAELA/Photos and videos/Animate testing"
CAPUCHIN_LOG_FILE = "D:/MIKAELA/Photos and videos/Animate testing/capuchin_detected_images.txt"
CAPUCHIN_TRAIN_DIR = "D:/MIKAELA/Photos and videos/Animate training"

# Detection settings
CAPUCHIN_CONFIDENCE_THRESHOLD = 0.15  # Set to 15%

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Load the trained capuchin model if available
capuchin_model = None
if os.path.exists(CAPUCHIN_MODEL_PATH):
    capuchin_model = tf.keras.models.load_model(CAPUCHIN_MODEL_PATH)
else:
    print(f"Warning: Capuchin model not found at {CAPUCHIN_MODEL_PATH}. Run `--retrain` first.")

def find_images_recursively(root_dir):
    """Recursively finds all images in the given directory and subdirectories."""
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(dirpath, filename))
    return image_paths

def process_capuchin_images(root_directory):
    """Process images, detect capuchins, and copy detected files."""
    ensure_directory_exists(CAPUCHIN_TARGET_DIR)
    detected_images = []

    if capuchin_model is None:
        print("Error: Capuchin model not loaded. Please run `--retrain` first.")
        return

    # Find all images recursively
    image_paths = find_images_recursively(root_directory)
    print(f"Found {len(image_paths)} images. Checking for capuchins...")

    for image_path in image_paths:
        try:
            # Load and preprocess image
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            # Run model prediction
            predictions = preprocess_and_predict(image_path)
            if not predictions:
                print(f"No predictions found for {image_path}. Skipping.")
                continue

            # Check for capuchin with confidence ≥ threshold
            for _, label, confidence in predictions:
                if "capuchin" in label.lower() and confidence >= CAPUCHIN_CONFIDENCE_THRESHOLD:
                    print(f"Capuchin detected in {image_path} with confidence {confidence:.2f}")

                    # Save image path to log
                    detected_images.append(image_path)

                    # Copy image to the target directory
                    destination_path = os.path.join(CAPUCHIN_TARGET_DIR, os.path.basename(image_path))
                    if not os.path.exists(destination_path):
                        shutil.copy(image_path, destination_path)
                    break  # Stop checking other predictions if capuchin is found
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # Save detected image paths to a text file
    with open(CAPUCHIN_LOG_FILE, "w") as f:
        f.write("\n".join(detected_images))

    print(f"Detection complete. {len(detected_images)} capuchin images copied to '{CAPUCHIN_TARGET_DIR}'.")
    print(f"File paths saved to '{CAPUCHIN_LOG_FILE}'.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze images or videos.")
    parser.add_argument("--type", type=str, choices=["images", "videos"], help="Analyze 'images' or 'videos'")
    parser.add_argument("--capuchin", action="store_true", help="Run capuchin detection mode")
    parser.add_argument("--retrain", action="store_true", help="Retrain the capuchin model")
    args = parser.parse_args()

    if args.retrain:
        retrain_model(CAPUCHIN_TRAIN_DIR, CAPUCHIN_MODEL_PATH)

    if args.capuchin:
        print("Scanning for Capuchins in the flash drive...")
        process_capuchin_images(CAPUCHIN_SOURCE_DIR)

    if args.type:
        if args.type == "images":
            image_directory = "C:/Users/buddh/Desktop/VS Code/ANIMATE/data/images"
            process_images_from_directory(image_directory)
        elif args.type == "videos":
            video_directory = "C:/Users/buddh/Desktop/VS Code/ANIMATE/data/videos"
            process_videos_from_directory(video_directory)
