import cv2
import numpy as np


def shift_intensity(img, r):
    return img + np.random.randint(-r, r)


def augment_slice_thickness(image, max_r=5):
    r = np.random.randint(1, max_r + 1)
    return np.expand_dims(cv2.resize(image[::r], image.shape[:2][::-1]), 2)


def random_occlusion(img, r=0.5, num=1):
    rank = 2
    s = img.shape
    out = img.copy()
    for i in range(num):

        example_size = np.random.randint(0, int(r * min(s)), (2,))
        valid_loc_range = [s[i] - example_size[i] for i in range(rank)]
        rnd_loc = [np.random.randint(valid_loc_range[dim])
                   if valid_loc_range[dim] > 0 else np.zeros(1, dtype=int) for dim in range(rank)]
        slicer = [slice(rnd_loc[i], rnd_loc[i] + example_size[i]) for i in range(rank)]

        if np.random.rand() > 0.5:
            out[slicer] = np.random.randint(img.min(), img.max())
        else:
            out[slicer] = (img.max() - img.min()) * np.random.rand(example_size[0], example_size[1]) + img.min()
    return out
