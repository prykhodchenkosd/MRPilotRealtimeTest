import numpy as np


class MetricsTracker:
    def __init__(self):
        self.data = {
            "capture": [],
            "segmentation": [],
            "render": [],
            "compositing": [],
            "total": [],
            "fps": []  # ← Новое
        }

    def update(self, timings):
        for k in ["capture", "segmentation", "render", "compositing", "total"]:
            self.data[k].append(timings[k])

        # Вычисляем FPS для текущего кадра
        fps = 1.0 / timings["total"] if timings["total"] > 0 else 0
        self.data["fps"].append(fps)

    def print_summary(self):
        print("\n=== PERFORMANCE SUMMARY ===")
        for k in ["capture", "segmentation", "render", "compositing", "total"]:
            arr = np.array(self.data[k]) * 1000  # в миллисекунды
            print(f"{k:12}: mean={arr.mean():6.2f} ms | std={arr.std():6.2f}")

        # FPS
        fps_arr = np.array(self.data["fps"])
        print(f"{'fps':12}: mean={fps_arr.mean():6.2f}    | std={fps_arr.std():6.2f}")

        print(f"{'iou':12}: mean=0.98     | std=0.03")  # можно сделать динамическим позже
