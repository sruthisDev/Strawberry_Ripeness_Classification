import os

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from skimage.feature import graycomatrix, graycoprops

WEBAPP_DIR = os.path.dirname(os.path.dirname(__file__))
app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_strawberry_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

LABELS = {0: "unripe", 1: "ripe"}


def compute_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0


def compute_aspect_ratio(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h > 0 else 0


def extract_features(image):
    image = cv2.resize(image, (256, 256))

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_h = hsv_image[:, :, 0].mean()
    mean_s = hsv_image[:, :, 1].mean()
    mean_v = hsv_image[:, :, 2].mean()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]

    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity, aspect_ratio = 0, 0
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        circularity = compute_circularity(largest_contour)
        aspect_ratio = compute_aspect_ratio(largest_contour)

    vector = [[mean_h, mean_s, mean_v, contrast, homogeneity, energy, circularity, aspect_ratio]]

    readout = {
        "hue_degrees": round(float(mean_h) * 2, 1),
        "saturation_pct": round(float(mean_s) / 255 * 100, 1),
        "value_pct": round(float(mean_v) / 255 * 100, 1),
        "texture_contrast": round(float(contrast), 2),
        "texture_homogeneity": round(float(homogeneity), 3),
        "circularity": round(float(circularity), 3),
        "aspect_ratio": round(float(aspect_ratio), 3),
    }

    return vector, readout


@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    features, readout = extract_features(image)
    features_normalized = scaler.transform(features)

    prediction = svm_model.predict(features_normalized)[0]
    probabilities = svm_model.predict_proba(features_normalized)[0]
    confidence = float(probabilities[int(prediction)])

    return jsonify({
        "label": LABELS[int(prediction)],
        "confidence": round(confidence, 4),
        "features": readout,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
