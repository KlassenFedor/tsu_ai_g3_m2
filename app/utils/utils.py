from skimage.transform import resize
import numpy as np
from skimage import measure
import cv2


def apply_brightness_contrast(input_img, brightness=0, contrast=0, prod=False):
    brightness = map_c(brightness, 0, 510, -255, 255)
    contrast = map_c(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def map_c(x, in_min, in_max, out_min, out_max):
    return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


def get_contours_count_with_model(image, model, sample_size, classes):
    sample = apply_brightness_contrast(image, 100 + 255, 65 + 127, prod=False)
    sample = resize(sample, sample_size)
    predict = model.predict(sample.reshape((1,) + sample_size + (3,)))
    predict = predict.reshape(sample_size + (classes,))
    contours = measure.find_contours(np.array(predict[:, :, 0]))

    return len(contours)


def get_biggest_area_with_model(image, model, sample_size, classes):
    sample = apply_brightness_contrast(image, 100 + 255, 65 + 127, prod=False)
    sample = resize(sample, sample_size)
    predict = model.predict(sample.reshape((1,) + sample_size + (3,)))
    predict = predict.reshape(sample_size + (classes,))
    contours = measure.find_contours(np.array(predict[:, :, 0]))

    biggest_area = 0
    for contour in contours:
        c = np.expand_dims(contour.astype(np.float32), 1)
        c = cv2.UMat(c)
        area = cv2.contourArea(c)

        if area > biggest_area:
            biggest_area = area

    return biggest_area


def get_contours_count(image, sample_size, classes):
    sample = resize(image, sample_size)
    sample = sample.reshape(sample_size + (classes,))
    contours = measure.find_contours(np.array(sample[:, :, 0]))

    return len(contours)


def get_biggest_area(image, sample_size, classes):
    sample = resize(image, sample_size)
    sample = sample.reshape(sample_size + (classes,))
    contours = measure.find_contours(np.array(sample[:, :, 0]))

    biggest_area = 0
    for contour in contours:
        c = np.expand_dims(contour.astype(np.float32), 1)
        c = cv2.UMat(c)
        area = cv2.contourArea(c)

        if area > biggest_area:
            biggest_area = area

    return biggest_area