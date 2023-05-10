from skimage.transform import resize
import numpy as np
from skimage import measure
import cv2


def get_contours_count(image, model, sample_size, classes):
    sample = resize(image, sample_size)
    predict = model.predict(sample.reshape((1,) + sample_size + (3,)))
    predict = predict.reshape(sample_size + (classes,))
    contours = measure.find_contours(np.array(predict[:, :, 0]))

    return len(contours)


def get_biggest_area(image, model, sample_size, classes):
    sample = resize(image, sample_size)
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