# Anti-Spoofing Face Recognition

This repository contains code and resources for an anti-spoofing face recognition project. The project includes scripts for data collection, training, and testing a face-recognition model with defenses against spoofing (printed photos, replay attacks, etc.).

IMPORTANT: keep your own local copies of the following folders in the project root and do NOT commit them to public repositories:

- `Dataset/` — your image dataset (e.g. `Real/`, `Fake/`, `all/`).
- `known_faces/` — enrolled user images or serialized face encodings used at inference time.
- `models/` — trained model weights and checkpoints (PyTorch `.pt` files).
- `myenv/` — optional Python virtual environment folder used locally.

These folders contain private data, trained weights and environment files. The code expects them to exist at runtime. If they're missing, create them and populate with your own data and models.

## Repository layout

- `dataCollection.py` — capture and label face images (real vs fake).
- `splitData.py` — split dataset into train/val/test sets.
- `train.py` — training script for the anti-spoofing/recognition model.
- `main.py` — example/demo script for inference (detection + recognition + anti-spoofing).
- `debug.py` — helper/debug utilities.
- `yoloTest.py` — scripts and tests for YOLO-based detection.
- `models/` — trained model files (local; keep your own here).
- `Dataset/` — dataset folder (local; keep your own here).
- `known_faces/` — enrolled faces and encodings (local; keep your own here).
- `myenv/` — virtual environment (local; optional).
- `yolov8n.pt`, `yolov8l.pt` — included detection weights (may be used for face detection).

## Requirements

- Python 3.8+ (3.10/3.11 recommended)
- PyTorch (match your CUDA/CPU setup)
- OpenCV (cv2)
- face-recognition (dlib) or another embedding library (insightface, facenet)
- ultralytics (if using YOLOv8 for detection)
- numpy, tqdm, scikit-learn

If a `requirements.txt` file is not present, install packages manually. Example (PowerShell):

```powershell
# create venv (optional) and activate
python -m venv myenv
.\myenv\Scripts\Activate.ps1

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python face-recognition ultralytics numpy tqdm scikit-learn
```

Adjust the PyTorch wheel URL to match your CUDA version or install CPU-only wheels if necessary.

## Quick start

1. Prepare dataset
   - Place your images under `Dataset/` (for example `Dataset/Real/` and `Dataset/Fake/`). Follow the repository's existing structure if present.
2. Split data
   - Run `splitData.py` to create training and test splits.
3. Train
   - Configure hyperparameters inside `train.py` as needed and run:

```powershell
python train.py
```

Training will save checkpoints under `models/` and logs under `runs/` by default.

4. Inference / Demo
   - Ensure `known_faces/` contains enrolled users (images or encoding files).
   - Run the main demo:

```powershell
python main.py
```

`main.py` typically runs face detection, computes embeddings, compares to known faces, and applies anti-spoofing logic. Inspect the script to tune thresholds and paths.

## Files to inspect

- `models/model.py` — model architecture.
- `train.py` — training loop and checkpointing.
- `dataCollection.py` — dataset capture and labeling.
- `splitData.py` — dataset splitting.

## Best practices

- Keep private data (face images, known face encodings, and environment folders) out of version control. Use `.gitignore` to exclude `Dataset/`, `known_faces/`, `models/`, and `myenv/`.
- Store only the smallest necessary metadata or synthetic examples in the repo.
- When sharing results, prefer saving anonymized embeddings or evaluation summaries rather than raw images.

## Troubleshooting

- Faces not detected: verify the path to the YOLO model (`yolov8n.pt` or `yolov8l.pt`) and make sure detection thresholds are set correctly.
- Slow training: check for GPU availability and that your PyTorch installation supports CUDA. Use smaller batch sizes if you run out of memory.
- Mismatched devices: ensure model weights and code are on the same device (CPU vs GPU).

## Next steps I can help with

- Add a `requirements.txt` generated from imports.
- Create a `config.yaml` for paths, thresholds, training hyperparameters.
- Add a small `check_setup.py` script that verifies existence of `Dataset/`, `known_faces/`, `models/`, and `myenv/` and prints helpful messages.

