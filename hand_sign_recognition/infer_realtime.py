import argparse
import json
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from utils.landmark_extractor import HandLandmarkExtractor

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/hand_sign_model.h5")
    p.add_argument("--labels", type=str, default="models/labels.json")
    p.add_argument("--max-hands", type=int, default=1)
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--smooth-k", type=int, default=5, help="window smoothing size")
    p.add_argument("--draw", type=str, default="True", choices=["True", "False"])
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise RuntimeError("Model tidak ditemukan. Latih model dulu via train.py")
    if not os.path.exists(args.labels):
        raise RuntimeError("File labels.json tidak ditemukan.")

    with open(args.labels, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

    model = tf.keras.models.load_model(args.model)
    extractor = HandLandmarkExtractor(max_hands=args.max_hands)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak ditemukan.")

    probs_smooth = deque(maxlen=args.smooth_k)
    pred_label = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            landmarks, annotated = extractor.process(frame, draw=(args.draw == "True"))

            display = annotated.copy()
            if landmarks is not None:
                x = landmarks.reshape(1, -1)
                probs = model.predict(x, verbose=0)[0]
                probs_smooth.append(probs)
                probs_avg = np.mean(np.stack(probs_smooth, axis=0), axis=0)
                conf = float(np.max(probs_avg))
                idx = int(np.argmax(probs_avg))
                label = label_map.get(idx, str(idx))

                if conf >= args.min_conf:
                    pred_label = f"{label} ({conf:.2f})"
                else:
                    pred_label = "Unknown"

                bar_w = int(300 * conf)
                cv2.rectangle(display, (10, 80), (10 + bar_w, 105), (0, 255, 0), -1)

            cv2.putText(display, f"Pred: {pred_label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("Hand Sign Inference", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()

if __name__ == "__main__":
    main()
