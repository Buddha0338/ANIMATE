import time
import matplotlib.pyplot as plt
import numpy as np

processing_times = []  # Collect processing times
elephant_confidences = []  # Collect elephant confidence levels

def process_images_from_directory(directory, filter_species=None):
    reference_histograms = load_reference_histograms()
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png"))]
    grouped_images = group_images_by_time(image_paths)

    for image_group in grouped_images:
        species_detected = process_image_group(image_group)
        for image_path in image_group:
            start_time = time.time()  # Start time
            processed_image = preprocess_image(image_path)
            predictions = preprocess_and_predict(image_path)
            end_time = time.time()  # End time

            # Collect processing time
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            # Collect elephant confidence levels
            for pred in predictions:
                if 'elephant' in pred[1].lower():
                    elephant_confidences.append(pred[2])

    # Compute and display processing time statistics
    avg_time = np.mean(processing_times)
    median_time = np.median(processing_times)
    std_time = np.std(processing_times)

    print(f"Average Time: {avg_time:.4f}s, Median Time: {median_time:.4f}s, Std Time: {std_time:.4f}s")

    # Create scatter plot for processing times
    plt.scatter(range(len(processing_times)), processing_times)
    plt.title("Processing Times for Images")
    plt.xlabel("Image Index")
    plt.ylabel("Processing Time (s)")
    plt.show()

    # Compute and display confidence statistics
    avg_confidence = np.mean(elephant_confidences)
    print(f"Average Confidence Level for Elephants: {avg_confidence:.2f}")

    # Create histogram for elephant confidence levels
    plt.hist(elephant_confidences, bins=10, range=(0, 1), alpha=0.75)
    plt.title("Elephant Confidence Levels")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.show()
