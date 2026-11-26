# tests for classifier.py, it's input should be landmarks and output sign and confidence
import random
from types import SimpleNamespace
from classifier import predict, LABELS


def make_fake_hand(seed=0):
    """Build 21 fake landmarks like MediaPipe's (each has .x, .y, .z, 21 pts per hand)."""
    random.seed(seed)
    return [
        SimpleNamespace(x=random.random(), y=random.random(), z=random.random())
        for _ in range(21)
    ]


# affirm that the LABElS arr contains the proper info
def test_labels_are_digits_and_letters():
    assert len(LABELS) == 36
    assert LABELS[0] == "0"
    assert LABELS[9] == "9"
    assert LABELS[10] == "A"
    assert LABELS[35] == "Z"


def test_predict_returns_label_and_confidence():
    result = predict(make_fake_hand())
    assert "label" in result
    assert "confidence" in result


def test_label_is_a_valid_class():
    assert predict(make_fake_hand())["label"] in LABELS


def test_confidence_is_between_0_and_1():
    assert 0.0 <= predict(make_fake_hand())["confidence"]
    assert predict(make_fake_hand())["confidence"] <= 1.0


def test_same_input_does_same_output():
    hand = make_fake_hand(seed=1)
    assert predict(hand) == predict(hand)


def test_hand_size_and_position_does_not_matter():
    hand = make_fake_hand(seed=2)
    moved_and_bigger = [
        SimpleNamespace(x=p.x * 3 + 5, y=p.y * 3 - 2, z=p.z * 3 + 1) for p in hand
    ]
    assert predict(moved_and_bigger)["label"] == predict(hand)["label"]


def test_zero_size_hand_does_not_crash():
    flat = [SimpleNamespace(x=0.5, y=0.5, z=0.5) for _ in range(21)]
    assert predict(flat)["label"] in LABELS
