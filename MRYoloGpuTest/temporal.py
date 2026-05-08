# ============================================
# TEMPORAL STABILITY METRICS
# ============================================

import numpy as np


class TemporalMetrics:

    def __init__(self):

        self.prev_mask = None

    def compute_iou(
        self,
        current_mask
    ):

        current = current_mask[:, :, 0] > 0.5

        if self.prev_mask is None:
            self.prev_mask = current
            return 1.0

        intersection = np.logical_and(
            self.prev_mask,
            current
        ).sum()

        union = np.logical_or(
            self.prev_mask,
            current
        ).sum()

        iou = intersection / (union + 1e-6)

        self.prev_mask = current

        return float(iou)