# reads ASL sign via output from hand_tracking.py
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

LABELS = list("0123456789") + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# access asl-landmark-mlp (landmark to ASL sign converter) from huggingface
path = hf_hub_download("nocontextdoruk/asl-landmark-mlp", "mlp_asl.onnx")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def predict(landmarks):
    # 21 pts to numbers, centered on the wrist and scaled by hand size
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
    pts = pts - pts[0]
    pts = pts / (np.linalg.norm(pts[9]) or 1.0)

    # run model
    logits = session.run(None, {"input": pts.reshape(1, 63)})[0][0]

    # turn raw scores into percentages and pick the best one
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    best = int(probs.argmax())

    return {"label": LABELS[best], "confidence": round(float(probs[best]), 4)}
