# Screen Reader

A real-time screen reader supporting:

- **Saliency**: spectral residual saliency to highlight major objects
- **Face detection**:
  - **Haar Cascade** with CLAHE preprocessing (lightweight fallback)
  - **DNN** (ResNet-SSD) for robust detection
  - **MediaPipe** for real-time detection with CLAHE & optimized settings
  - **Ensemble (face_combo)**: union of DNN, MediaPipe, and Haar for maximal recall

---

## Installation

1. **Create & activate a virtual environment** (highly recommended)

   **Unix / macOS**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

### DNN Model Setup

Download the Caffe model and prototxt into `screen_reader/models/`:

```bash
mkdir -p screen_reader/models
curl -L -o screen_reader/models/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
curl -L -o screen_reader/models/res10_300x300_ssd_iter_140000.caffemodel \
  https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

## Usage

### Saliency detection

```bash
python -m screen_reader.main --monitor 2 --box-method saliency --live --threshold 128 --min-area 500
```

### Haar Cascade + CLAHE Face Detection

```bash
python -m screen_reader.main --monitor 2 --box-method face_haar --live --scale 1.05 --min-neighbors 3
```

### DNN Face Detection (with Haar fallback)

```bash
python -m screen_reader.main --monitor 2 --box-method face_dnn --live --conf-thresh 0.5
```

### MediaPipe Face Detection

```bash
python -m screen_reader.main --monitor 2 --box-method face_mp --live --mp-min-conf 0.3
```

### Ensemble Face Detection

```bash
python -m screen_reader.main --monitor 2 --box-method face_combo --live \
  --conf-thresh 0.5 --mp-min-conf 0.3 --scale 1.05 --min-neighbors 3
```

---

## Advanced Options & Debugging

- **Suppress warnings and info messages:**  
  Add `--suppress-warnings` to hide most terminal warnings and TensorFlow/MediaPipe info.

- **Enable debug output:**  
  Add `--debug` to print detection confidences and details for each frame.

- **Log detection results to a file:**  
  Add `--log-file detections.txt` to save detection results for later analysis.

- **Show FPS in the window title:**  
  Add `--show-fps` to display frames per second in the window title.

**Examples:**

```bash
# MediaPipe with debug output and FPS
python -m screen_reader.main --monitor 1 --box-method face_mp --live --debug --show-fps

# MediaPipe with suppressed warnings and logging
python -m screen_reader.main --monitor 1 --box-method face_mp --live --suppress-warnings --log-file mp_log.txt

# MediaPipe with a custom confidence threshold
python -m screen_reader.main --monitor 1 --box-method face_mp --live --mp-min-conf 0.5
```

---

## Configuration Parameters

- `--box-method`: `saliency`, `face_haar`, `face_dnn`, `face_mp`, or `face_combo`
- Saliency: `--threshold`, `--min-area`
- Haar: `--scale`, `--min-neighbors`
- DNN: `--conf-thresh`
- MediaPipe: `--mp-min-conf`
- Debugging: `--debug`, `--log-file`, `--show-fps`, `--suppress-warnings`
