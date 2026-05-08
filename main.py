import cv2
import time
import numpy as np

from segmentation import get_mask
from renderer import render_virtual_scene
from compositing import composite
from metrics import MetricsTracker

cap = cv2.VideoCapture(0)
metrics = MetricsTracker()

print("Starting MR prototype... Press 'q' to exit.")

while True:
    t_start = time.time()

    # --- 1. Capture ---
    ret, frame = cap.read()
    if not ret:
        break
    t_capture = time.time()

    # --- 2. Segmentation ---
    mask = get_mask(frame)
    t_seg = time.time()

    # --- 3. Rendering ---
    virtual = render_virtual_scene(frame.shape)
    t_render = time.time()

    # --- 4. Compositing ---
    output = composite(frame, virtual, mask)
    t_comp = time.time()

    # --- 5. Metrics ---
    metrics.update({
        "capture": t_capture - t_start,
        "segmentation": t_seg - t_capture,
        "render": t_render - t_seg,
        "compositing": t_comp - t_render,
        "total": t_comp - t_start
    })

    # --- Visualization ---
    cv2.imshow("Input", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Output", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

metrics.print_summary()

cap.release()
cv2.destroyAllWindows()