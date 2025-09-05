import argparse
import csv
import os

import cv2

from utils.landmark_extractor import HandLandmarkExtractor

def parse_args():
    p = argparse.ArgumentParser(description="Collect hand sign landmarks to CSV per label.")
    p.add_argument("--classes", type=str, required=True,
                   help="Comma-separated labels, e.g. A,B,C or ThumbsUp,OK")
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--output-dir", type=str, default="data/raw")
    p.add_argument("--max-hands", type=int, default=1)
    p.add_argument("--flip", type=str, default="True", choices=["True", "False"],
                   help="Flip horizontal to augment left/right hand")
    p.add_argument("--draw", type=str, default="True", choices=["True", "False"])
    return p.parse_args()

def main():
    args = parse_args()
    labels = [s.strip() for s in args.classes.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak ditemukan.")

    extractor = HandLandmarkExtractor(max_hands=args.max_hands)

    label_idx = 0
    per_label_count = {lbl: 0 for lbl in labels}

    print("Controls: [TAB] ganti label, [SPACE] ambil sampel, [Q] keluar.")
    print("Label urutan:", labels)
    print("Label aktif:", labels[label_idx])

    csv_files = {lbl: open(os.path.join(args.output_dir, f"{lbl}.csv"), "a", newline="") for lbl in labels}
    csv_writers = {lbl: csv.writer(csv_files[lbl]) for lbl in labels}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.flip == "True":
                frame = cv2.flip(frame, 1)

            landmarks, annotated = extractor.process(frame, draw=(args.draw == "True"))

            hud = annotated.copy()
            cv2.putText(hud, f"Label: {labels[label_idx]}  Count: {per_label_count[labels[label_idx]]}/{args.samples_per_class}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(hud, "TAB: Ganti Label | SPACE: Capture | Q: Quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Collect Data", hud)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and landmarks is not None:
                lbl = labels[label_idx]
                csv_writers[lbl].writerow(landmarks.tolist())
                per_label_count[lbl] += 1

            if key == 9:  # TAB
                label_idx = (label_idx + 1) % len(labels)
                print("Label aktif:", labels[label_idx])

            if key == ord('q'):
                break

            if all(per_label_count[lbl] >= args.samples_per_class for lbl in labels):
                print("Selesai: jumlah sampel terpenuhi.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        for f in csv_files.values():
            f.close()

if __name__ == "__main__":
    main()
