import cv2
import numpy as np
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon, polygon_perimeter
from skimage import measure
import imutils


class VideoHandler:
    def __init__(self):
        self.rgb_colors = [
            (255, 255, 255),
            (255, 99, 71)
        ]

    def frame_to_mask(self, frame, input_size, output_size, classes, model):
        sample = resize(frame, input_size)

        predict = model.predict(sample.reshape((1,) + input_size + (3,)))
        predict = predict.reshape(input_size + (classes,))

        scale = frame.shape[0] / input_size[0], frame.shape[1] / input_size[1]
        frame = (frame / 1.5).astype(np.uint8)

        biggest_area = 0
        contours_count = 0

        for channel in range(0, classes):
            contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
            contours = measure.find_contours(np.array(predict[:, :, channel]))

            biggest_area = 0
            contours_count = len(contours)
            for contour in contours:
                c = np.expand_dims(contour.astype(np.float32), 1)
                c = cv2.UMat(c)
                area = cv2.contourArea(c)

                if area > biggest_area:
                    biggest_area = area

                rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                           contour[:, 1] * scale[1],
                                           shape=contour_overlay.shape)

                contour_overlay[rr, cc] = 1

            contour_overlay = dilation(contour_overlay, disk(1))
            frame[contour_overlay == 1] = self.rgb_colors[channel]

            for contour in contours:
                c = np.expand_dims(contour.astype(np.float32), 1)
                c = cv2.UMat(c)
                area = cv2.contourArea(c)

                if area == biggest_area:
                    rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                               contour[:, 1] * scale[1],
                                               shape=contour_overlay.shape)

                    contour_overlay[rr, cc] = 2

            contour_overlay = dilation(contour_overlay, disk(1))
            frame[contour_overlay == 2] = self.rgb_colors[1]

        return frame, contours_count, np.round(biggest_area / (output_size[0] * output_size[1]), 6)

    def apply_brightness_contrast(self, input_img, brightness=0, contrast=0, prod=False):
        brightness = map(brightness, 0, 510, -255, 255)
        contrast = map(contrast, 0, 254, -127, 127)

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

        if not prod:
            cv2.putText(buf, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        return buf

    def map(self, x, in_min, in_max, out_min, out_max):
        return int((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def show_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video stream or file")

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = self.apply_brightness_contrast(frame, 100 + 255, 65 + 127, prod=False)

                frame = imutils.resize(frame, width=700)
                frame, contours, area = self.frame_to_mask(frame)
                cv2.putText(frame, 'Amount:{}'.format(contours), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            1)
                cv2.putText(frame, 'Max:{}'.format(area), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('Frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        cap.release()
        cv2.destroyAllWindows()
