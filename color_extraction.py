import pandas as pd

from image_utils import iter_labeled_images, compute_hsv_means, append_features_csv

# Input directories with labels
input_dirs = {
    "Data/Pickable": 1,      # Label 1 for ripe fruit
    "Data/UnPickable": 0     # Label 0 for unripe fruit
}

output_file = "fruit_hsv_features.csv"

features_list = []
for filename, label, image in iter_labeled_images(input_dirs):
    mean_h, mean_s, mean_v = compute_hsv_means(image)
    features_list.append([filename, mean_h, mean_s, mean_v, label])

df_new = pd.DataFrame(features_list, columns=["Filename", "Mean_H", "Mean_S", "Mean_V", "Label"])

append_features_csv(df_new, output_file)
print(f"HSV feature extraction complete. Data saved to {output_file}")
