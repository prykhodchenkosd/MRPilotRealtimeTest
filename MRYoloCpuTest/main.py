# ============================================
# REAL-TIME MR REGION SUBSTITUTION SYSTEM
# ============================================

import cv2
import time
import torch
from segmentation import YOLOSegmentation
from renderer import VirtualRenderer
from compositing import RegionCompositor
from metrics import PerformanceTracker
from temporal import TemporalMetrics


# ============================================
# INITIALIZATION
# ============================================
print(torch.cuda.is_available())

segmentor = YOLOSegmentation(
    model_path="yolov8n-seg.pt",
    device="cpu"
    #device="cuda"
)

renderer = VirtualRenderer(
    image_path="sky.jpg"
)

compositor = RegionCompositor()

tracker = PerformanceTracker()

temporal_metrics = TemporalMetrics()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("MR system started.")
print("Press ESC to exit.")

# ============================================
# MAIN LOOP
# ============================================
frame_id = 0

previous_mask = None
previous_segmentation_view = None
previous_detections = None
while True:

    t0 = time.time()

    # ----------------------------------------
    # 1. CAPTURE
    # ----------------------------------------

    ret, frame = cap.read()

    if not ret:
        break

    t1 = time.time()

    # ----------------------------------------
    # 2. SEGMENTATION
    # ----------------------------------------

    if frame_id % 3 == 0:

        # Run heavy segmentation

        segmentation_view, mask, detections = (
            segmentor.process(frame)
        )

        # Save results

        previous_mask = mask
        previous_segmentation_view = segmentation_view
        previous_detections = detections

    else:

        # Reuse previous results

        mask = previous_mask
        segmentation_view = previous_segmentation_view
        detections = previous_detections

    t2 = time.time()
    # ----------------------------------------
    # 3. VIRTUAL RENDERING
    # ----------------------------------------

    virtual_scene = renderer.render(
        frame.shape
    )

    t3 = time.time()

    # ----------------------------------------
    # 4. COMPOSITING
    # ----------------------------------------

    output = compositor.composite(
        frame,
        virtual_scene,
        mask
    )

    t4 = time.time()

    # ----------------------------------------
    # 5. TEMPORAL STABILITY
    # ----------------------------------------

    temporal_iou = temporal_metrics.compute_iou(
        mask
    )

    # ----------------------------------------
    # 6. PERFORMANCE METRICS
    # ----------------------------------------

    total_time = t4 - t0

    fps = 1.0 / (total_time + 1e-6)

    tracker.update({
        "capture": (t1 - t0) * 1000,
        "segmentation": (t2 - t1) * 1000,
        "render": (t3 - t2) * 1000,
        "compositing": (t4 - t3) * 1000,
        "total": total_time * 1000,
        "fps": fps,
        "iou": temporal_iou
    })

    # ----------------------------------------
    # 7. VISUALIZATION
    # ----------------------------------------

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"Temporal IoU: {temporal_iou:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Input",
        frame
    )

    cv2.imshow(
        "Segmentation",
        segmentation_view
    )

    cv2.imshow(
        "MR Output",
        output
    )

    key = cv2.waitKey(1)
    frame_id += 1
    if key == 27:
        break


# ============================================
# CLEANUP
# ============================================

cap.release()

cv2.destroyAllWindows()

tracker.summary()