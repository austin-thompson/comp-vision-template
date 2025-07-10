import argparse
import cv2
from .capture import grab_screen
from .processor import (
    detect_saliency_boxes,
    detect_face_haar,
    detect_face_dnn,
    detect_face_mp
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
    return p.parse_args()

def main():
    args = parse_args()

    def process_frame(frame):
        if args.box_method == "saliency":
            boxes = detect_saliency_boxes(frame, args.threshold, args.min_area)
        elif args.box_method == "face_haar":
            boxes = detect_face_haar(frame, args.scale, args.min_neighbors)
        elif args.box_method == "face_dnn":
            dnn_boxes = detect_face_dnn(frame, args.conf_thresh)
            boxes = dnn_boxes if dnn_boxes else detect_face_haar(frame, args.scale, args.min_neighbors)
        elif args.box_method == "face_mp":
            boxes = detect_face_mp(frame, args.mp_min_conf)
        else:  # face_combo ensemble
            boxes = detect_face_dnn(frame, args.conf_thresh)
            boxes.update(detect_face_mp(frame, args.mp_min_conf))
            boxes.update(detect_face_haar(frame, args.scale, args.min_neighbors))

        for _, (x, y, w, h) in boxes.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame

    window_name = f"ScreenReader - {args.box_method}"
    if args.live:
        while True:
            frame = grab_screen(args.monitor)
            disp = process_frame(frame)
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

if __name__ == "__main__":
    main()
