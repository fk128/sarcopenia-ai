import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import pearsonr

from sarcopenia_ai.apps.slice_detection.dataloader import image_slide_generator
from sarcopenia_ai.preprocessing.preprocessing import pad_image_to_size
from sarcopenia_ai.preprocessing.preprocessing import reduce_hu_intensity_range
from sarcopenia_ai.utils import extract_mip


def get_best_loc(loc, height=100, step=1):
    d = height
    s = d - np.array(list(range(0, d, step)))

    max_v = 0
    max_i = 0
    for i in range(len(loc) - d):
        v = pearsonr(loc[i:i + d], s)[0]
        if v > max_v:
            max_v = v
            max_i = i
    return int(max_i + height / 2)


def predict_reg(model, image, y, input_shape, start=0, step=1):
    gen = image_slide_generator(image, y, input_shape, start=start, step=step)
    loc = []
    loc_abs = []
    weights = []
    height = input_shape[0]
    mid = height // 2
    for i, (image_batch, y_batch) in enumerate(gen):
        preds = model.predict(image_batch[:, :, :, :])
        v = int(preds[0])
        t = y_batch + start + step * i
        loc.append(v)
        # if preds[1] > 0.5:
        # if v > 0 or v < height:
        loc_abs.append(v + start + step * i)
        if len(preds) == 2:
            weights.append(preds[1])
        else:
            weights.append(1)

    # if dual output
    if len(preds) == 2:
        p = np.dot(np.squeeze(np.array(loc_abs)), np.squeeze(np.array(weights))) / np.sum(weights)
    else:
        i_best = get_best_loc(loc, step=step)
        try:
            p = loc_abs[i_best]
        except:
            p = np.mean(loc_abs)
    # avg_pred0 = int(sum(np.array(weights) * np.array(loc_abs)) / sum(weights))
    # avg_pred0 = int(np.array(loc_abs[i_best + height // 3:i_best + 2 * height // 3]).mean())
    return int(p), 1.0  # prob 1 as no prob value


def find_max(img):
    return np.unravel_index(np.argmax(img, axis=None), img.shape)[0]


def preprocess_mip_for_slice_detection(image, spacing, target_spacing, min_height=512,
                                       min_width=512):
    image = zoom(image, [spacing[2] / target_spacing, spacing[0] / target_spacing])
    image = reduce_hu_intensity_range(image)

    v = min_height if image.shape[0] <= min_height else 2 * min_height
    img_size = [v, min_width]
    padded_image = pad_image_to_size(image, img_size, loc=[1, -1], mode='constant')
    padded_image = padded_image[:v, :min_width] - 128
    return padded_image[np.newaxis, :, :, np.newaxis], image


def preprocess_sitk_image_for_slice_detection(sitk_image, target_spacing=1, mode='frontal', min_height=512,
                                              min_width=512):
    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    dx = int(direction[0])
    dy = int(direction[4])
    dz = int(direction[8])

    image = sitk.GetArrayFromImage(sitk_image)[::dx, ::dy, ::dz]

    image_frontal, image_sagittal = extract_mip(image)

    if mode == 'sagittal':
        image = image_sagittal
    else:
        image = image_frontal

    return preprocess_mip_for_slice_detection(image, spacing, target_spacing, min_height,
                                              min_width)


def decode_slice_detection_prediction(preds):
    max_z = find_max(preds[0, :])
    prob = float(preds.max())

    return max_z, prob


def adjust_detected_position_spacing(z, spacing, offset=0):
    return int(z // spacing[2] + offset)


def place_line_on_img(img, y, pred, r=2):
    if len(img.shape) == 2 or img.shape[2] != 3:
        img = np.dstack([img] * 3)
    v = img.max()
    img[pred - r:pred + r, :, 0] = 0.5 * v
    img[y - r:y + r, :, 1] = 0.5 * v
    return img


def preprocess_test_image(img):
    height = 512
    width = 512
    if img.shape[0] <= height:
        v = height
    else:
        v = 2 * height
    img_size = [v, width]
    img = pad_image_to_size(img, img_size, loc=[1, -1], mode='constant')
    return img[:v, :width] - 128


def predict_slice(model, img, ds):
    img = preprocess_test_image(img)
    img = img[np.newaxis, :, :, np.newaxis]
    preds = model.predict(img)

    m = ds * find_max(preds[0, :]) + ds // 2
    max_pred = preds.max()
    return m, max_pred, preds, img
