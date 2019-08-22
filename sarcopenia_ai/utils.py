import numpy as np


def extract_mask(img, lower, upper):
    mask1 = img > lower
    mask2 = img < upper
    return np.logical_and(mask1, mask2).astype(np.float32)


def compute_muscle_attenuation(image, segmentation):
    masked = image[segmentation > 0]
    return np.mean(masked)


def compute_muscle_area(mask, spacing, units='mm'):
    area = np.sum(mask) * np.prod(spacing)
    if units == 'mm':
        return area
    elif units == 'cm':
        return area / 100
    else:
        raise ValueError('Invalid area unit.')


def extract_mip(image, d=10, s=40):
    image_c = image.copy()

    image_c[:, :s, ] = 0
    image_c[:, -s:, ] = 0
    image_c[:, :, :s] = 0
    image_c[:, :, -s:] = 0

    (_, _, Z) = np.meshgrid(range(image.shape[1]), range(image.shape[0]), range(image.shape[2]))
    M = Z * (image_c > 0)
    M = M.sum(axis=2) / (image_c > 0).sum(axis=2)
    M[np.isnan(M)] = 0
    mask = M > 0
    c = int(np.mean(M[mask]))

    image_frontal = np.max(image_c, axis=1)

    image_sagittal = np.max(image_c[:, :, c - d:c + d], axis=2)[::-1, :]

    return image_frontal, image_sagittal
