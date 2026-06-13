import os, json
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "models/cnn_model.h5"
CLASS_INDICES = "class_indices.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model & class mapping
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INDICES, "r") as f:
    class_indeces = json.load(f)
idx_to_class = {v: k for k, v in class_indeces.items()}

def prepare_image(path, target=(64,64)):
    img = Image.open(path).convert("RGB").resize(target)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    x = prepare_image(save_path)
    probs = model.predict(x)[0]
    top_idx = np.argmax(probs)

    result = {
        "class": idx_to_class[top_idx],
        "probability": float(probs[top_idx]),
        "image_url": f"/{save_path}"
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
