import numpy as np


def pad_to_square(img):
    h, w, _ = img.shape
    pad = int((w - h) / 2)

    img_pad = np.zeros((w, w, 3), dtype=np.uint8)
    img_pad[pad:w-pad, :, :] = img

    return img_pad, pad