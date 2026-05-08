import cv2
import numpy as np

virtual_img = cv2.imread("sky.jpg")

def render_virtual_scene(shape):
    h, w, _ = shape

    resized = cv2.resize(virtual_img, (w, h))
    return resized.astype(np.float32) / 255.0