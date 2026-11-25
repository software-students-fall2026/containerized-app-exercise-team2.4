from pathlib import Path

from hand_tracking import detect_hand


def main():
    image_path = Path("test_hand.jpg")

    if not image_path.exists():
        print("Please add test_hand.jpg first.")
        return

    image_bytes = image_path.read_bytes()

    landmarks = detect_hand(image_bytes)

    if landmarks is None:
        print("No hand was detected.")
        return

    print("Hand detected!")
    print("Number of landmarks:", len(landmarks))

    for index, point in enumerate(landmarks):
        print(
            f"Point {index}: "
            f"x={point.x:.3f}, "
            f"y={point.y:.3f}, "
            f"z={point.z:.3f}"
        )


if __name__ == "__main__":
    main()
