from UNet import UNet
from app.utils.video_handler import VideoHandler
import fire
import logging
logging.basicConfig(level=logging.DEBUG, filename="../py_log.log", filemode="a")


class Model:
    def __init__(self):
        self.structure = UNet()
        self.video_handler = None

    def train(self, dataset):
        logging.info('Train started')

        self.structure.set_data(dataset)
        self.structure.fit(20)

        logging.info('Train ended successfully')

    def evaluate(self, dataset):
        logging.info('Evaluation started')

        self.structure.set_data(dataset)
        res = self.structure.unet.evaluate(self.structure.test_dataset.take(10))

        logging.info('Evaluation ended successfully, with metrics: iou={}, dice={}'.format(res[1], res[2]))

        return res

    def demo(self, video_path):
        logging.info('Demonstration started')

        self.structure.load_weights('SemanticSegmentationLesson/networks/unet_like_aug_v2')
        self.video_handler = VideoHandler(self.structure.unet)
        self.video_handler.show_video(video_path)

        logging.info('Demonstration ended successfully')


if __name__ == '__main__':
    fire.Fire(Model)
