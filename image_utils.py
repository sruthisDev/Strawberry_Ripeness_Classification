import os
import cv2
import pandas as pd

IMAGE_EXTENSIONS = (".jpg", ".png")


def iter_labeled_images(input_dirs, size=(256, 256)):
    """Yield (filename, label, resized_image) for every image in each labeled directory."""
    for input_dir, label in input_dirs.items():
        for filename in os.listdir(input_dir):
            if filename.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)
                image = cv2.resize(image, size)
                yield filename, label, image


def compute_hsv_means(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_h = hsv_image[:, :, 0].mean()
    mean_s = hsv_image[:, :, 1].mean()
    mean_v = hsv_image[:, :, 2].mean()
    return mean_h, mean_s, mean_v


def append_features_csv(df_new, output_file):
    """Append newly extracted features to output_file, creating it if needed."""
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_csv(output_file, index=False)
