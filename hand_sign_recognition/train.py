import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PROC_DIR = "data/processed"
MODELS_DIR = "models"

def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    data_path = os.path.join(PROC_DIR, "dataset.npz")
    labels_path = os.path.join(PROC_DIR, "labels.json")

    if not os.path.exists(data_path):
        raise RuntimeError("dataset.npz tidak ditemukan. Jalankan preprocess.py dulu.")

    data = np.load(data_path)
    X, y = data["X"], data["y"]
    num_classes = int(y.max()) + 1

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "hand_sign_model.h5"),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks
    )

    with open(labels_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(os.path.join(MODELS_DIR, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("Model saved to models/hand_sign_model.h5")
    print("Labels saved to models/labels.json")

if __name__ == "__main__":
    main()
