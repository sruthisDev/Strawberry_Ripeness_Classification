import cv2
import os
import numpy as np

# Input and Output Directories
input_dir = "Data/Sample"
output_dir = "processed_images/sample"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process all images
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read image
        image = cv2.imread(os.path.join(input_dir, filename))

        # Resize
        image = cv2.resize(image, (256, 256))

        # Noise Reduction
        image = cv2.GaussianBlur(image, (5,5), 0)

        # Convert to HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Background Segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save Processed Image
        cv2.imwrite(os.path.join(output_dir, filename), image)

print("Batch Processing Complete!")
