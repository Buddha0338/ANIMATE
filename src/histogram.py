import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Function to calculate histogram for an image
def calculate_histogram(image):
    histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return histogram.flatten()

# Function to load images from a folder and calculate histograms
def load_images_and_calculate_histograms(folder_path):
    histograms = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            histogram = calculate_histogram(image)
            histograms.append(histogram)
    return histograms

# Define paths
day_images_folder = 'C:/Users/buddh/Desktop/VS Code/ANIMATE/data/histogram/day'
night_images_folder = 'C:/Users/buddh/Desktop/VS Code/ANIMATE/data/histogram/night'

# Calculate histograms for day and night images
histograms_day = load_images_and_calculate_histograms(day_images_folder)
histograms_night = load_images_and_calculate_histograms(night_images_folder)

# Create reference histograms by averaging the histograms of day and night images
reference_histogram_day = np.mean(histograms_day, axis=0)
reference_histogram_night = np.mean(histograms_night, axis=0)

# Save reference histograms
with open('C:/Users/buddh/Desktop/VS Code/ANIMATE/src/reference_histograms.pkl', 'wb') as f:
    pickle.dump({
        'day': reference_histogram_day,
        'night': reference_histogram_night
    }, f)

# Optional: Visualize reference histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Day Histogram")
plt.plot(reference_histogram_day)
plt.xlim([0, 512])

plt.subplot(1, 2, 2)
plt.title("Night Histogram")
plt.plot(reference_histogram_night)
plt.xlim([0, 512])

plt.show()
