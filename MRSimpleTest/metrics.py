import numpy as np

class MetricsTracker:
    def __init__(self):
        self.data = {
            "capture": [],
            "segmentation": [],
            "render": [],
            "compositing": [],
            "total": []
        }

    def update(self, timings):
        for k in self.data:
            self.data[k].append(timings[k])

    def print_summary(self):
        print("\n=== PERFORMANCE SUMMARY ===")
        for k in self.data:
            arr = np.array(self.data[k]) * 1000  # ms
            print(f"{k}: mean={arr.mean():.2f} ms | std={arr.std():.2f}")
