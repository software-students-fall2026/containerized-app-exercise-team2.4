# communication between hands_tracking.py and classifer.py
from flask import Flask, jsonify, request
from classifier import predict
from hand_tracking import detect_hand

app = Flask(__name__)

# any confidence level below this classified as an "unsure" guess
CONFIDENCE_THRESHOLD = 0.6


@app.post("/predict")
def predict_sign():
    # get uploaded frame
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "missing 'image' file"}), 400

    # find the 21 landmarks + interpret into ASL letter/num
    landmarks = detect_hand(file.read())
    if landmarks is None:
        return jsonify({"label": None, "confidence": 0.0, "reason": "no_hand"})
    result = predict(landmarks)

    # if not confident enough
    if result["confidence"] < CONFIDENCE_THRESHOLD:
        return jsonify(
            {
                "label": None,
                "confidence": result["confidence"],
                "reason": "low_confidence",
            }
        )

    return jsonify(result)


# checks if server is running
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=8000)
