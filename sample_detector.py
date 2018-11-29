import logging
import os

import torch
from torch.autograd import Variable
from yolov3.darknet import Darknet
from yolov3.preprocess import prep_frame
from yolov3.util import load_classes, write_results

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

BASE_PATH = os.path.join(os.path.dirname(__file__))


class Detector:
    cfg = os.path.join(BASE_PATH, 'cfg', 'yolov3.cfg')
    weights = os.path.join(BASE_PATH, 'weights', 'yolov3.weights')

    def __init__(self):
        super().__init__()
        self.class_names = load_classes(os.path.join(BASE_PATH, 'data', 'coco.names'))
        self.num_classes = 80
        self.batch_size = 1
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.scales = "1,2,3"
        self.inp_dim = 416  # k*32 where k is int and >1
        self.cuda = torch.cuda.is_available()
        self._load_model()

    def _load_model(self):
        model = Darknet(Detector.cfg)
        model.load_weights(Detector.weights)
        model.net_info["height"] = str(self.inp_dim)
        if self.cuda:
            model.cuda()
        model.eval()
        self.model = model

    def get_localization(self, image):
        ret = []
        im_batch = prep_frame(image, self.inp_dim)
        im_dim_list = [image.shape[1], image.shape[0]]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        if self.cuda:
            im_dim_list = im_dim_list.cuda()
            batch = im_batch.cuda()

        with torch.no_grad():
            prediction = self.model(Variable(batch), self.cuda)

        output = write_results(prediction, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)

        if type(output) == int:
            return ret

        im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

        scaling_factor = torch.min(self.inp_dim / im_dim_list, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        output = output.int().cpu().numpy()
        torch.cuda.empty_cache()

        return output


if __name__ == "__main__":
    from moviepy.video.io.VideoFileClip import VideoFileClip
    import time
    import pickle
    import random
    import cv2


    def pipeline(img):
        ret = detector.get_localization(img)

        for x in ret:
            c1 = (x[1], x[2])
            c2 = (x[3], x[4])
            cls = x[-1]
            label = "{0}".format(detector.class_names[cls])
            color = random.choice(colors)
            cv2.rectangle(img, c1, c2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    detector = Detector()
    colors = pickle.load(open("pallete", "rb"))
    output = 'test_yolo.mp4'
    start = time.time()
    clip1 = VideoFileClip("../../car25_compressed.mp4").subclip(180, 200)
    clip = clip1.fl_image(pipeline)
    clip.write_videofile(output, audio=False)
    end = time.time()
    print(round(end - start, 2), 'Seconds to finish')
