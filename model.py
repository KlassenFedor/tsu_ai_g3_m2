from UNet import UNet
from video_handler import VideoHandler
import fire
import glob
from skimage.io import imread, imsave, imshow
from utils import get_contours_count, get_biggest_area


class Model:
    def __init__(self, trained_model):
        self.structure = UNet()
        self.video_handler = VideoHandler(trained_model)

    def train(self, dataset):
        self.structure.set_data(dataset)
        self.structure.fit(20)

    def evaluate(self, dataset):
        count_mse = 0
        area_mse = 0

        masks = sorted(glob.glob(dataset + 'masks/*.jpg'))
        images = sorted(glob.glob(dataset + 'images/*.jpg'))
        for i in range(len(images)):
            file = imread(images[i])
            mask = imread(masks[i])

            real_count = get_contours_count(mask, self.structure.SAMPLE_SIZE, 1)
            real_area = get_biggest_area(mask, self.structure.SAMPLE_SIZE, 1)
            predicted_count = get_contours_count(file, self.structure, self.structure.SAMPLE_SIZE, 1)
            predicted_area = get_biggest_area(file, self.structure, self.structure.SAMPLE_SIZE, 1)

            count_mse += (real_count - predicted_count) ** 2
            area_mse += (real_area - predicted_area) ** 2

        return count_mse / len(images), area_mse / len(images)

    def demo(self, video_path):
        self.video_handler.show_video(video_path)


if __name__ == '__main__':
    fire.Fire(Model)
