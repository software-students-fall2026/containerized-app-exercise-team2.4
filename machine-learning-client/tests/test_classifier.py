"""Test the complete ASL hand detection and classification pipeline."""

from pathlib import Path
from hand_tracking import detect_hand
from classifier import predict


def main():
    image_path = Path("test_hand.jpg")

    if not image_path.exists():
        print("Please add test_hand.jpg to the machine-learning-client folder.")
        return

    image_bytes = image_path.read_bytes()

    # detect hand landmarks from the image.
    landmarks = detect_hand(image_bytes)

    if landmarks is None:
        print("No hand was detected.")
        return

    print("Hand detected")
    print("Number of landmarks:", len(landmarks))

    # classify the detected hand landmarks.
    result = predict(landmarks)

    print("Predicted ASL sign:", result["label"])
    print(f'Confidence: {result["confidence"] * 100:.2f}%')


if __name__ == "__main__":
    main()
