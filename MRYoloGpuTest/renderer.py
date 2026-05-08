# ============================================
# VIRTUAL SCENE RENDERER
# ============================================

import cv2
import numpy as np
import os


class VirtualRenderer:

    def __init__(self, image_path="assets/virtual_scene.jpg"):

        # ------------------------------------
        # TRY LOAD IMAGE
        # ------------------------------------

        if os.path.exists(image_path):

            self.virtual_scene = cv2.imread(image_path)

            if self.virtual_scene is None:

                print(
                    "[WARNING] Failed to load image."
                )

                self.virtual_scene = (
                    self.generate_fallback_scene()
                )

            else:

                print(
                    "[INFO] Virtual scene loaded."
                )

        else:

            print(
                "[WARNING] Virtual scene not found."
            )

            print(
                "[INFO] Using fallback synthetic scene."
            )

            self.virtual_scene = (
                self.generate_fallback_scene()
            )

    # ========================================
    # FALLBACK SYNTHETIC SCENE
    # ========================================

    def generate_fallback_scene(self):

        h = 720
        w = 1280

        scene = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky
        scene[:h // 2] = (255, 180, 80)

        # Ground
        scene[h // 2:] = (50, 180, 50)

        # Horizon
        cv2.line(
            scene,
            (0, h // 2),
            (w, h // 2),
            (255, 255, 255),
            3
        )

        # Runway
        cv2.rectangle(
            scene,
            (w // 2 - 100, h // 2),
            (w // 2 + 100, h),
            (80, 80, 80),
            -1
        )

        return scene

    # ========================================
    # RENDER
    # ========================================

    def render(self, shape):

        h, w, _ = shape

        scene = cv2.resize(
            self.virtual_scene,
            (w, h)
        )

        scene = (
            scene.astype(np.float32) / 255.0
        )

        return scene