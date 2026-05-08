# ============================================
# PERFORMANCE METRICS
# ============================================

import numpy as np


class PerformanceTracker:

    def __init__(self):

        self.data = {
            "capture": [],
            "segmentation": [],
            "render": [],
            "compositing": [],
            "total": [],
            "fps": [],
            "iou": []
        }

    def update(self, values):

        for k in values:
            self.data[k].append(values[k])

    def summary(self):

        print("\n=== PERFORMANCE SUMMARY ===")

        for k, v in self.data.items():

            arr = np.array(v)

            print(
                f"{k}: "
                f"mean={arr.mean():.2f} "
                f"| std={arr.std():.2f}"
            )