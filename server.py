from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from detector import analyze_currency


BASE_DIR = Path(__file__).resolve().parent
STATIC_FILES = {"index.html", "styles.css", "app.js"}

app = Flask(__name__, static_folder=None)


def read_image_bytes(data: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path: str):
    if path in STATIC_FILES:
        return send_from_directory(BASE_DIR, path)
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze_api():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"error": "No image file was uploaded."}), 400

    image = read_image_bytes(uploaded.read())
    if image is None:
        return jsonify({"error": "The uploaded file is not a readable image."}), 400

    return jsonify(analyze_currency(image))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
