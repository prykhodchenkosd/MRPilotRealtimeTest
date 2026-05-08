import numpy as np

def composite(real, virtual, mask):
    real = real.astype(np.float32) / 255.0

    output = (1 - mask) * real + mask * virtual
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return output
