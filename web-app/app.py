"""Flask web app for ASL alphabet picture-to-text."""

import io
import math
import os
from datetime import datetime, timezone

import requests
from flask import Flask, current_app, render_template, request
from PIL import Image, UnidentifiedImageError
from pymongo import MongoClient
from pymongo.errors import PyMongoError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


def validate_image(image_bytes):
    """Check whether the uploaded file is a valid image."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
                return False

            if image.width * image.height > 20_000_000:
                return False

            image.verify()

        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ):
        return False


def create_app(test_config=None):
    """Build the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
    app.config["MONGO_URI"] = os.environ.get(
        "MONGO_URI", "mongodb://localhost:27017/app"
    )
    if test_config:
        app.config.update(test_config)
    if "MONGO_CLIENT" not in app.config:
        app.config["MONGO_CLIENT"] = MongoClient(
            app.config["MONGO_URI"],
            serverSelectionTimeoutMS=2000,
        )

    # confirm connection
    try:
        app.config["MONGO_CLIENT"].admin.command("ping")
        print("CONNECTED TO MONGODB SUCCESSFULLY!")
    except Exception as e:
        print(f"MONGODB CONNECTION FAILED: {e}")

    @app.get("/")
    def index():
        """Show the upload page and the newest translation."""
        return render_index()

    @app.post("/translate")
    def translate():
        """Validate and translate an uploaded handshape photo."""
        image = request.files.get("image")

        if image is None or not image.filename:
            return render_index(
                error="Choose an image first.",
                status=400,
            )

        filename = os.path.basename(image.filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            return render_index(
                error="Upload a JPG, PNG, WEBP, or GIF.",
                status=400,
            )

        image_bytes = image.read()

        if not image_bytes:
            return render_index(
                error="That file is empty.",
                status=400,
            )

        if not validate_image(image_bytes):
            return render_index(
                error="Invalid or corrupted image.",
                status=400,
            )

        result = classify_image(image_bytes, filename)

        if result.get("error"):
            return render_index(
                error=result["error"],
                status=result.get("status", 422),
            )

        saved = save_translation(filename, result)

        if saved is None:
            return render_index(
                error="Could not reach the database.",
                status=500,
            )

        return render_index(latest=saved)

    @app.get("/history")
    def history():
        """Show saved translations."""
        try:
            rows = load_history()
        except PyMongoError:
            return (
                render_template(
                    "history.html",
                    rows=[],
                    error="Could not reach the database.",
                ),
                500,
            )
        return render_template("history.html", rows=rows, error=None)

    @app.errorhandler(413)
    def too_large(_error):
        """Reject uploads over the size limit."""
        return render_index(
            error="Image must be 5 MB or smaller.",
            status=413,
        )

    return app


def render_index(error=None, latest=None, status=200):
    """Render the upload page."""
    if latest is None:
        latest = safe_latest()
    page = render_template("index.html", latest=latest, error=error)
    return page, status


def safe_latest():
    """Return the newest translation, or None if the database is down."""
    try:
        rows = load_history()
    except PyMongoError:
        return None
    if not rows:
        return None
    return rows[0]


def collection():
    """Return the translations collection."""
    mongo = current_app.config["MONGO_CLIENT"]
    return mongo["asltranslationapp"]["translations"]


def load_history():
    """Return recent translations, newest first."""
    cursor = collection().find().sort("created_at", -1).limit(50)
    return [serialize(document) for document in cursor]


def save_translation(filename, result):
    document = {
        "letter": result["letter"],
        "confidence": result["confidence"],
        "filename": filename,
        "created_at": datetime.now(timezone.utc),
        "error": result.get("error"),
    }

    try:
        collection().insert_one(document)
    except PyMongoError:
        return None

    return serialize(document)


def serialize(document):
    """Turn a Mongo document into template fields."""
    created = document.get("created_at")
    if isinstance(created, datetime):
        stamp = created.strftime("%Y-%m-%d %H:%M UTC")
    else:
        stamp = created or ""
    return {
        "letter": document.get("letter") or "?",
        "confidence_label": format_confidence(document.get("confidence")),
        "filename": document.get("filename") or "",
        "created_at": stamp,
        "error": document.get("error"),
    }


def format_confidence(value):
    """Format a valid confidence score as a percent."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"

    if not math.isfinite(number) or not 0 <= number <= 1:
        return "n/a"

    return f"{round(number * 100)}%"


def classify_image(image_bytes, filename):
    """Request and validate a prediction from the ML client."""
    url = os.environ.get(
        "ML_CLIENT_URL",
        "http://ml-client:5000/predict",
    )

    try:
        response = requests.post(
            url,
            files={
                "image": (
                    filename,
                    image_bytes,
                    "application/octet-stream",
                )
            },
            timeout=10,
        )

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            raise ValueError("Invalid response")

        messages = {
            "no_hand": "No hand detected.",
            "low_confidence": "Not sure. Try a clearer photo.",
        }

        reason = payload.get("reason")

        if reason is not None:
            if reason not in messages:
                raise ValueError("Unknown recognition error")

            return {
                "letter": "?",
                "confidence": 0.0,
                "error": messages[reason],
                "status": 422,
            }

        raw_letter = payload.get("letter") or payload.get("label")

        if not isinstance(raw_letter, str):
            raise ValueError("Missing letter")

        letter = raw_letter.strip().upper()

        if len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return {
                "letter": "?",
                "confidence": 0.0,
                "error": "No valid ASL letter recognized.",
                "status": 422,
            }

        raw_confidence = payload.get("confidence")

        if isinstance(raw_confidence, bool):
            raise ValueError("Invalid confidence")

        confidence = float(raw_confidence)

        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence")

        return {
            "letter": letter,
            "confidence": confidence,
            "error": None,
        }

    except requests.Timeout:
        return {
            "letter": "?",
            "confidence": 0.0,
            "error": "Recognition timed out. Please try again.",
            "status": 504,
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        AttributeError,
        OverflowError,
    ):
        return {
            "letter": "?",
            "confidence": 0.0,
            "error": "Recognizer is not available yet.",
            "status": 502,
        }


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000)
