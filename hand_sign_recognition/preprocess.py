import os
import glob
import json
import numpy as np

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise RuntimeError("Tidak ada file CSV di data/raw/. Jalankan collect_data.py dulu.")

    X_list, y_list, labels = [], [], []
    for idx, f in enumerate(files):
        label = os.path.splitext(os.path.basename(f))[0]
        labels.append(label)
        data = np.loadtxt(f, delimiter=",", dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        X_list.append(data)
        y_list.append(np.full((data.shape[0],), idx, dtype=np.int64))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    rng = np.random.default_rng(123)
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    np.savez(os.path.join(OUT_DIR, "dataset.npz"), X=X, y=y)
    with open(os.path.join(OUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({i: lbl for i, lbl in enumerate(labels)}, f, ensure_ascii=False, indent=2)

    print("Saved:", os.path.join(OUT_DIR, "dataset.npz"))
    print("Saved:", os.path.join(OUT_DIR, "labels.json"))

if __name__ == "__main__":
    main()
