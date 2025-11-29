import os
import tempfile
import numpy as np
import tifffile as tiff
import tensorflow as pd  # specific import not needed if loading model directly, but keeping for env consistency
from tensorflow.keras.models import load_model
from skimage.transform import resize
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64

app = Flask(__name__)

# Configuration
IMG_SIZE = (64, 64)
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
MODEL_PATH = "../models/model.h5"

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- PREPROCESSING  ---
def process_input(file_path):
    # Read 13-band image
    img = tiff.imread(file_path)

    # Resize to (64, 64, 13)
    img_resized = resize(
        img, (IMG_SIZE[0], IMG_SIZE[1], 13), preserve_range=True, anti_aliasing=True
    )

    # Normalizations
    img_resized = img_resized.astype(np.float32)
    if img_resized.max() > 0:
        img_resized /= img_resized.max()

    # Add batch dimension: (1, 64, 64, 13)
    return np.expand_dims(img_resized, axis=0), img_resized


# Visualization helper
def get_rgb_view(img_array):
    """
    Extracts RGB channels for display. Blue=1, Green=2, Red=3
    """
    rgb = img_array[:, :, [3, 2, 1]]

    # Normalize just for display brightness
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    return rgb


# --- ROUTES ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    tmp_path = None
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Process Input
        input_tensor, raw_img = process_input(tmp_path)

        # Generate Prediction
        if model:
            probs = model.predict(input_tensor)[0]
            top_idx = np.argmax(probs)
            top_class = CLASSES[top_idx]
            confidence = float(probs[top_idx])

            # Convert probabilities to list for JSON
            prob_list = probs.tolist()
        else:
            # Dummy response if model is missing
            top_class = "Model Not Loaded"
            confidence = 0.0
            prob_list = [0.1] * 10

        # Generate RGB Image for Frontend
        rgb_img = get_rgb_view(raw_img)

        # Convert 0-1 float array to 0-255 uint8 for PNG conversion
        rgb_uint8 = (rgb_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_uint8)

        # Save to buffer
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        img_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

        return jsonify(
            {
                "class": top_class,
                "confidence": confidence,
                "image_b64": img_b64,
                "probabilities": prob_list,
                "classes": CLASSES,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    app.run(debug=True)
