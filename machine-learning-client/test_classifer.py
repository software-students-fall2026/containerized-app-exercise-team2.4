"""Tests for classifier.py"""

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


def test_labels_are_digits_and_letters():
    """affirm that the LABElS arr contains the proper info"""
    assert len(LABELS) == 36
    assert LABELS[0] == "0"
    assert LABELS[9] == "9"
    assert LABELS[10] == "A"
    assert LABELS[35] == "Z"


def test_predict_returns_label_and_confidence():
    """affirm that predict returns a label and confidence value"""
    result = predict(make_fake_hand())
    assert "label" in result
    assert "confidence" in result


def test_label_is_a_valid_class():
    """affirm that the prediction label is valid (A-Z 0-9)"""
    assert predict(make_fake_hand())["label"] in LABELS


def test_confidence_is_between_0_and_1():
    """affirm confidence value is valid (between 0 to 1)"""
    assert 0.0 <= predict(make_fake_hand())["confidence"]
    assert predict(make_fake_hand())["confidence"] <= 1.0


def test_same_input_does_same_output():
    """affirm that predictions are consistent, same input results in same output"""
    hand = make_fake_hand(seed=1)
    assert predict(hand) == predict(hand)


def test_hand_size_and_position_does_not_matter():
    """affirm that the size/position of hand does not affect prediction"""
    hand = make_fake_hand(seed=2)
    moved_and_bigger = [
        SimpleNamespace(x=p.x * 3 + 5, y=p.y * 3 - 2, z=p.z * 3 + 1) for p in hand
    ]
    assert predict(moved_and_bigger)["label"] == predict(hand)["label"]


def test_zero_size_hand_does_not_crash():
    """affirm that a divide by zero error does not occur when all points are identical"""
    flat = [SimpleNamespace(x=0.5, y=0.5, z=0.5) for _ in range(21)]
    assert predict(flat)["label"] in LABELS
