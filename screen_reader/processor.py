import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe face detection setup with optimized parameters
_mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.3
)

def detect_saliency_boxes(img, threshold=128, min_area=500):
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, sal_map = sal.computeSaliency(img)
    sal_map = (sal_map * 255).astype('uint8')
    _, th = cv2.threshold(sal_map, threshold, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            dets.append((x, y, w, h))
    return {i: box for i, box in enumerate(dets)}

def detect_face_haar(img, scale=1.1, min_neighbors=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    return {i: tuple(f) for i, f in enumerate(faces)}

# Lazy-loaded DNN for face_dnn
_dnn_net = None
def detect_face_dnn(img, conf_thresh=0.5):
    global _dnn_net
    if _dnn_net is None:
        proto = os.path.join(os.path.dirname(__file__), 'models', 'deploy.prototxt')
        model = os.path.join(os.path.dirname(__file__), 'models', 'res10_300x300_ssd_iter_140000.caffemodel')
        if not os.path.isfile(proto) or not os.path.isfile(model):
            raise FileNotFoundError(f'DNN model files not found. Expected: {proto} and {model}')
        _dnn_net = cv2.dnn.readNetFromCaffe(proto, model)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    _dnn_net.setInput(blob)
    detections = _dnn_net.forward()
    boxes = {}
    idx = 0
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf >= conf_thresh:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes[idx] = (x1, y1, x2 - x1, y2 - y1)
            idx += 1
    return boxes

def detect_face_mp(img, mp_min_conf=0.3):
    # Pre-equalize image using CLAHE on LAB L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Run MediaPipe detection
    results = _mp_face.process(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB))
    h, w = img.shape[:2]
    boxes = {}
    if results.detections:
        for i, det in enumerate(results.detections):
            conf = det.score[0]
            if conf < mp_min_conf:
                continue
            bbox = det.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            boxes[i] = (x1, y1, x2 - x1, y2 - y1)
    return boxes
