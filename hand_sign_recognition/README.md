# Hand Sign Language Recognition (Python 3.11)

Proyek ini mengenali *static hand signs* (posisi tangan diam) menggunakan:
- **MediaPipe** untuk ekstraksi 21 landmark tangan (x, y).
- **TensorFlow/Keras** untuk klasifikasi.
- **OpenCV** untuk input webcam dan tampilan.

> Catatan: Proyek ini fokus pada *static sign*. Untuk *dynamic sign* (gerakan berurutan), Anda bisa memperluasnya dengan menambah penampung urutan (sequence) dan model RNN/Temporal.

## Persyaratan
- Python **3.11**
- Webcam

## Instalasi (Windows/Mac/Linux)
```bash
# 1) Buat virtualenv
python3.11 -m venv .venv
# Aktifkan:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 2) Clone/copy project ini, lalu:
pip install --upgrade pip
pip install -r requirements.txt
```

## Alur Kerja
1. **Koleksi Data**: ambil data untuk tiap kelas (label) via webcam.
2. **Preprocess**: gabungkan semua CSV jadi dataset terproses.
3. **Train**: latih model dan simpan ke folder `models/`.
4. **Inferensi Realtime**: jalankan prediksi via webcam.

## 1) Koleksi Data
Jalankan:
```bash
python collect_data.py --classes A,B,C --samples-per-class 200
```
- Tekan tombol yang sesuai label untuk merekam satu sampel:
  - `A` = label 0, `B` = label 1, `C` = label 2 (dan seterusnya sesuai urutan `--classes`).
- Tekan `SPACE` untuk ambil otomatis satu sampel ke label yang sedang dipilih.
- Tekan `TAB` untuk ganti label aktif.
- Tekan `Q` untuk keluar.
- Data akan tersimpan per label di `data/raw/<label>.csv`.

Opsi tambahan:
```bash
python collect_data.py --classes A,B,C --max-hands 1 --flip True --draw True
```

## 2) Preprocess
Menggabungkan semua CSV dari `data/raw/` menjadi satu dataset:
```bash
python preprocess.py
```
Hasil:
- `data/processed/dataset.npz` (X, y)
- `data/processed/labels.json` (pemetaan index→label)

## 3) Train
Latih model Dense kecil:
```bash
python train.py --epochs 30 --batch-size 64
```
Hasil:
- `models/hand_sign_model.h5`
- `models/labels.json` (copy dari processed)

## 4) Inferensi Realtime
Jalankan webcam + prediksi:
```bash
python infer_realtime.py --min-conf 0.6 --max-hands 1 --draw True
```

## Struktur Proyek
```
hand_sign_recognition/
├── data/
│   ├── raw/               # CSV per label dari koleksi data
│   └── processed/         # dataset.npz + labels.json
├── models/                # model terlatih (.h5) + labels.json
├── utils/
│   ├── __init__.py
│   └── landmark_extractor.py
├── collect_data.py
├── preprocess.py
├── train.py
├── infer_realtime.py
└── requirements.txt
```

## Tips Kualitas Data
- Pastikan pencahayaan cukup dan background kontras.
- Ambil variasi orientasi, jarak, dan tangan kiri/kanan (aktifkan `--flip True` saat koleksi).
- Ambil >150–300 sampel per kelas untuk hasil lebih stabil.

## Lisensi
MIT
