import cv2
import time
import numpy as np
from segmentation import get_mask
from renderer import render_virtual_scene
from compositing import composite
from metrics import MetricsTracker

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
"""
for i in range(-20, 0):
    cap.set(cv2.CAP_PROP_EXPOSURE, i)
    print(i, cap.get(cv2.CAP_PROP_EXPOSURE))
    time.sleep(0.5)
"""
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#cap.set(cv2.CAP_PROP_FPS, 30)           

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   
cap.set(cv2.CAP_PROP_EXPOSURE, -8)       
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)       
cap.set(cv2.CAP_PROP_FOCUS, 0)           
cap.set(cv2.CAP_PROP_AUTO_WB, 0)         
metrics = MetricsTracker()

print("Starting MR prototype... Press 'q' to exit.")

frame_count = 0

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

    frame_count += 1

    # --- Visualization ---
    cv2.imshow("Input", frame)
    cv2.imshow("Mask", (mask * 255).astype(np.uint8))
    cv2.imshow("Output", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

metrics.print_summary()
print(f"\nTotal frames processed: {frame_count}")

cap.release()
cv2.destroyAllWindows()
