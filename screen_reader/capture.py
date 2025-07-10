import cv2
import numpy as np
from mss import mss

def grab_screen(monitor_index=1):
    with mss() as sct:
        monitors = sct.monitors
        if monitor_index >= len(monitors):
            raise ValueError(f"Monitor {monitor_index} not found (only {len(monitors)-1} available)")
        mon = monitors[monitor_index]
        sct_img = sct.grab(mon)
        arr = np.array(sct_img)
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return img
