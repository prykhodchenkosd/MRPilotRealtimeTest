# ============================================
# REGION-LEVEL COMPOSITING
# ============================================

import numpy as np


class RegionCompositor:

    def __init__(self):
        pass

    def composite(
        self,
        real_frame,
        virtual_frame,
        mask
    ):

        real = real_frame.astype(np.float32) / 255.0

        output = (
            (1.0 - mask) * real
            + mask * virtual_frame
        )

        output = np.clip(
            output * 255,
            0,
            255
        ).astype(np.uint8)

        return output