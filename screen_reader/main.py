import argparse
import cv2
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow INFO and WARNING
os.environ["CUDA_VISIBLE_DEVICES"] = (
    ""  # (Optional) Hide CUDA warnings if not using GPU
)

import sys
import warnings

warnings.filterwarnings("ignore")  # Suppress Python warnings

from .capture import grab_screen
from .processor import (
    detect_face_dnn,
    detect_face_haar,
    detect_face_mp,
    detect_saliency_boxes,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", type=int, default=1)
    p.add_argument(
        "--box-method",
        choices=["saliency", "face_haar", "face_dnn", "face_mp", "face_combo"],
        default="saliency",
    )
    p.add_argument("--live", action="store_true")
    # Saliency
    p.add_argument("--threshold", type=int, default=128)
    p.add_argument("--min-area", type=int, default=500)
    # Haar
    p.add_argument("--scale", type=float, default=1.05)
    p.add_argument("--min-neighbors", type=int, default=3)
    # DNN
    p.add_argument("--conf-thresh", type=float, default=0.5)
    # MediaPipe
    p.add_argument("--mp-min-conf", type=float, default=0.3)
    # Suppress warnings
    p.add_argument(
        "--suppress-warnings",
        action="store_true",
        help="Suppress warnings and stderr output",
    )
    # Debug mode
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for detection details",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log detection results (optional)",
    )
    p.add_argument(
        "--show-fps",
        action="store_true",
        help="Display FPS in the window title",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.suppress_warnings:
        warnings.filterwarnings("ignore")
        sys.stderr = open(os.devnull, "w")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def process_frame(frame):
        if args.box_method == "saliency":
            # Wrap saliency boxes in the tuple format for consistency
            boxes = [
                (box, "Saliency", None)
                for box in detect_saliency_boxes(frame, args.threshold, args.min_area)
            ]
        elif args.box_method == "face_haar":
            boxes = detect_face_haar(frame, args.scale, args.min_neighbors)
        elif args.box_method == "face_dnn":
            dnn_boxes = detect_face_dnn(frame, args.conf_thresh)
            boxes = (
                dnn_boxes
                if dnn_boxes
                else detect_face_haar(frame, args.scale, args.min_neighbors)
            )
        elif args.box_method == "face_mp":
            boxes = detect_face_mp(frame, args.mp_min_conf, debug=args.debug)
        else:  # face_combo ensemble
            boxes = []
            boxes += detect_face_dnn(frame, args.conf_thresh)
            boxes += detect_face_mp(frame, args.mp_min_conf)
            boxes += detect_face_haar(frame, args.scale, args.min_neighbors)

        for i, (box, method, conf) in enumerate(boxes):
            x, y, w, h = box
            label = f"{method}"
            if conf is not None:
                label += f" {conf:.2f}"
            label += f" #{i+1}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
            )
        return frame

    window_name = f"ScreenReader - {args.box_method}"
    last_time = time.time()
    frame_count = 0
    if args.live:
        while True:
            frame = grab_screen(args.monitor)
            disp = process_frame(frame)
            frame_count += 1
            if args.show_fps and frame_count % 10 == 0:
                now = time.time()
                fps = 10 / (now - last_time)
                last_time = now
                window_name = f"ScreenReader - {args.box_method} - FPS: {fps:.2f}"
            cv2.imshow(window_name, disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        frame = grab_screen(args.monitor)
        disp = process_frame(frame)
        cv2.imshow(window_name, disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.log_file:
        with open(args.log_file, "a") as f:
            for i, (box, method, conf) in enumerate(boxes):
                f.write(f"{method},{conf},{box}\n")


if __name__ == "__main__":
    main()


def detect_face_mp(img, mp_min_conf=0.3, debug=False):
    # ... detection code ...
    if debug:
        print(f"MediaPipe raw confidence: {conf}")
    # ... rest of function ...
