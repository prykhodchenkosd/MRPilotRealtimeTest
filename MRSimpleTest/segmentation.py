import cv2
import numpy as np

def get_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Пример: выделяем СИНЮЮ область (как "окно")
    lower = np.array([100, 100, 100])
    upper = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Сглаживание
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Нормализация [0..1]
    mask = mask.astype(np.float32) / 255.0

    # Приведение к 3 каналам
    mask = np.stack([mask]*3, axis=-1)

    return mask
