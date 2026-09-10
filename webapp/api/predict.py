import os

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from skimage.feature import graycomatrix, graycoprops

WEBAPP_DIR = os.path.dirname(os.path.dirname(__file__))
app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Pretrained ImageNet gate model - catches non-strawberry photos (people, objects,
# rooms) before the ripe/unripe model runs. See scripts/export_gate_model.py.
GATE_SESSION = ort.InferenceSession(os.path.join(MODEL_DIR, "gate_model.onnx"))
STRAWBERRY_CLASS_INDEX = 949  # torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1 "strawberry"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODELS = {
    "svm": {
        "model": joblib.load(os.path.join(MODEL_DIR, "svm_strawberry_model.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
    },
    "knn": {
        "model": joblib.load(os.path.join(MODEL_DIR, "knn_strawberry_model.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "knn_scaler.pkl")),
    },
    "ensemble": {
        "model": joblib.load(os.path.join(MODEL_DIR, "ensemble_strawberry_model.pkl")),
        "scaler": joblib.load(os.path.join(MODEL_DIR, "ensemble_scaler.pkl")),
    },
}

LABELS = {0: "unripe", 1: "ripe"}


def looks_like_strawberry(image, top_k=5):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    normalized = (resized - IMAGENET_MEAN) / IMAGENET_STD
    input_tensor = normalized.transpose(2, 0, 1)[None, ...].astype(np.float32)

    logits = GATE_SESSION.run(None, {"input": input_tensor})[0][0]
    top_indices = np.argsort(logits)[::-1][:top_k]
    return STRAWBERRY_CLASS_INDEX in top_indices


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
        "contour_found": bool(contours),
    }

    return vector, readout


@app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")


@app.route("/samples/<path:filename>")
def samples(filename):
    return send_from_directory(os.path.join(WEBAPP_DIR, "samples"), filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file in request"}), 400

    model_name = request.form.get("model", "svm")
    if model_name not in MODELS:
        return jsonify({"error": f"Unknown model '{model_name}'. Choose from: {', '.join(MODELS)}"}), 400

    file = request.files["image"]
    if file.mimetype not in ("image/jpeg", "image/png"):
        return jsonify({"error": "Only JPG or PNG photos are supported."}), 400

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    features, readout = extract_features(image)

    if not readout.pop("contour_found"):
        return jsonify({
            "error": "Couldn't find a clear strawberry shape in this photo. Try a well-lit, close-up photo of a single strawberry."
        }), 422

    if not looks_like_strawberry(image):
        return jsonify({
            "error": "This doesn't look like a strawberry photo. Try a clear, close-up photo of a single strawberry."
        }), 422

    model = MODELS[model_name]["model"]
    scaler = MODELS[model_name]["scaler"]
    features_normalized = scaler.transform(features)

    prediction = model.predict(features_normalized)[0]
    probabilities = model.predict_proba(features_normalized)[0]
    confidence = float(probabilities[int(prediction)])

    return jsonify({
        "label": LABELS[int(prediction)],
        "confidence": round(confidence, 4),
        "model": model_name,
        "features": readout,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
