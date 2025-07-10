import numpy as np
import cv2
from screen_reader import processor


def test_detect_saliency_boxes_empty():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = processor.detect_saliency_boxes(img)
    assert isinstance(boxes, dict)
    assert len(boxes) == 0


def test_detect_face_haar_no_faces():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = processor.detect_face_haar(img)
    assert isinstance(results, list)
    assert len(results) == 0


def test_detect_face_dnn_no_faces():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    results = processor.detect_face_dnn(img, conf_thresh=0.99)
    assert isinstance(results, list)
    assert len(results) == 0


def test_detect_face_mp_no_faces():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    results = processor.detect_face_mp(img, mp_min_conf=0.99)
    assert isinstance(results, list)
    assert len(results) == 0
