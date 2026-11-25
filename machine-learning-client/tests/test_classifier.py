"""Manual integration test for the ASL classifier."""

from pathlib import Path
from classifier import predict
from hand_tracking import detect_hand


def main():
    """Detect and classify an ASL hand sign from a test image"""
    image_path = Path(__file__).parent / "test_hand.jpg"

    if not image_path.is_file():
        print(f"Test image not found: {image_path}")
        return

    landmarks = detect_hand(image_path.read_bytes())
    if landmarks is None:
        print("No hand was detected.")
        return

    print(f"Hand detected: {len(landmarks)} landmarks")
    result = predict(landmarks)
    print(f"Predicted ASL sign: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
