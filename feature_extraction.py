import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Define Input Folders and Corresponding Labels
input_dirs = {
    "Data/Pickable": 1,     # Folder with ripe images (Label = 1)
    "Data/UnPickable": 0    # Folder with unripe images (Label = 0)
}

output_file = "fruit_features.csv"
features_list = []

# Functions for Shape Features
def compute_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

def compute_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h > 0 else 0

# Process Images from All Folders
for input_dir, label in input_dirs.items():
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256, 256))

            # Color Features (HSV)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(hsv_image[:, :, 0])
            mean_s = np.mean(hsv_image[:, :, 1])
            mean_v = np.mean(hsv_image[:, :, 2])

            # Convert to Grayscale for Texture & Shape
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # GLCM Texture Features
            glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]

            # Shape Features (Contour Analysis)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            circularity, aspect_ratio = 0, 0  # Default values
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                circularity = compute_circularity(largest_contour)
                aspect_ratio = compute_aspect_ratio(largest_contour)

            # Store extracted features
            features_list.append([filename, label, mean_h, mean_s, mean_v, contrast, homogeneity, energy, circularity, aspect_ratio])

# Convert to DataFrame
df_new = pd.DataFrame(features_list, columns=["Filename", "Label", "Mean_H", "Mean_S", "Mean_V", "Contrast", "Homogeneity", "Energy", "Circularity", "Aspect_Ratio"])

# Append Data to CSV Instead of Overwriting
if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    df_final = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_final = df_new

df_final.to_csv(output_file, index=False)
print(f"Feature extraction complete. Data appended to {output_file}")
