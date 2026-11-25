"""HTTP API for ASL handshape recognition."""

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from classifier import predict
from hand_tracking import detect_hand

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

CONFIDENCE_THRESHOLD = 0.6


@app.post("/predict")
def predict_sign():
    """Recognize an ASL sign from an uploaded image."""
    file = request.files.get("image")

    if file is None:
        return jsonify({"error": "Missing image file"}), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    try:
        landmarks = detect_hand(image_bytes)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ):
        return jsonify({"error": "Invalid or corrupted image"}), 400

    if landmarks is None:
        return jsonify(
            {
                "label": None,
                "confidence": 0.0,
                "reason": "no_hand",
            }
        )

    result = predict(landmarks)

    if result["confidence"] < CONFIDENCE_THRESHOLD:
        return jsonify(
            {
                "label": None,
                "confidence": result["confidence"],
                "reason": "low_confidence",
            }
        )

    return jsonify(result)


@app.errorhandler(413)
def too_large(_error):
    """Reject oversized uploads."""
    return jsonify({"error": "Image must be 5 MB or smaller"}), 413


@app.get("/health")
def health():
    """Check whether the server is running."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
