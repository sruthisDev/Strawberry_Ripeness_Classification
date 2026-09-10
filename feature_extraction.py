import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

from image_utils import iter_labeled_images, compute_hsv_means, append_features_csv

input_dirs = {
    "Data/Pickable": 1,     # Ripe / pickable fruit
    "Data/UnPickable": 0    # Unripe / unpickable fruit
}

output_file = "fruit_features.csv"


def compute_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0


def compute_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h > 0 else 0


features_list = []
for filename, label, image in iter_labeled_images(input_dirs):
    mean_h, mean_s, mean_v = compute_hsv_means(image)

    # Grayscale for Texture & Shape
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

    features_list.append([filename, label, mean_h, mean_s, mean_v, contrast, homogeneity, energy, circularity, aspect_ratio])

df_new = pd.DataFrame(
    features_list,
    columns=["Filename", "Label", "Mean_H", "Mean_S", "Mean_V", "Contrast", "Homogeneity", "Energy", "Circularity", "Aspect_Ratio"],
)

append_features_csv(df_new, output_file)
print(f"Feature extraction complete. Data appended to {output_file}")
