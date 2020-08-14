
import ppdet.utils.checkpoint as checkpoint
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import parse_fetches
from ppdet.core.workspace import load_config, create
from paddle import fluid
import os
import cv2
import glob

from lib.trafficLightColor import estimate_label
from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info

import numpy as np


class VehicleDetector(object):

    def __init__(self):

        self.size = 608

        self.draw_threshold = 0.5

        self.cfg = load_config('configs/ppyolo.yml')

        self.place = fluid.CUDAPlace(
            0) if self.cfg.use_gpu else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

        self.model = create(self.cfg.architecture)

        self.bbox_results = []

        self.process = False

        self.init_params()

    def draw_bbox(self, image, gray, catid2name, bboxes, threshold):

        raw = image.copy()

        for dt in np.array(bboxes):

            catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
            if score < threshold:
                continue
                                       
            xmin, ymin, w, h = bbox
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmin + w)
            ymax = int(ymin + h)
            if catid == 10 and xmax - xmin > 5:
                roi = raw[ymin:ymax, xmin:xmax].copy()
                light_color = estimate_label(roi)
            else:
                roi = gray[ymin:ymax, xmin:xmax]
                light_color = '{:.3f}'.format(10/np.mean(roi))
            cv2.putText(image, light_color, (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)

        return image

    def init_params(self):

        startup_prog = fluid.Program()
        infer_prog = fluid.Program()
        with fluid.program_guard(infer_prog, startup_prog):
            with fluid.unique_name.guard():
                inputs_def = self.cfg['TestReader']['inputs_def']
                inputs_def['iterable'] = True
                feed_vars, loader = self.model.build_inputs(**inputs_def)
                test_fetches = self.model.test(feed_vars)
        infer_prog = infer_prog.clone(True)

        self.exe.run(startup_prog)
        if self.cfg.weights:
            checkpoint.load_params(self.exe, infer_prog, self.cfg.weights)

        extra_keys = ['im_info', 'im_id', 'im_shape']
        self.keys, self.values, _ = parse_fetches(
            test_fetches, infer_prog, extra_keys)
        dataset = self.cfg.TestReader['dataset']
        anno_file = dataset.get_anno()
        with_background = dataset.with_background
        use_default_label = dataset.use_default_label

        self.clsid2catid, self.catid2name = get_category_info(anno_file, with_background,
                                                              use_default_label)

        is_bbox_normalized = False
        if hasattr(self.model, 'is_bbox_normalized') and \
                callable(self.model.is_bbox_normalized):
            is_bbox_normalized = self.model.is_bbox_normalized()

        self.is_bbox_normalized = is_bbox_normalized

        self.infer_prog = infer_prog

    def process_img(self, img):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        shape = img.shape[:2]

        img = cv2.resize(img, (self.size, self.size))

        # RBG img [224,224,3]->[3,224,224]
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std

        img = img.astype('float32')
        img = np.expand_dims(img, axis=0)

        shape = np.expand_dims(np.array(shape), axis=0)
        im_id = np.zeros((1, 1), dtype=np.int64)

        return img, im_id, shape

    def detect(self, img, gray):

        self.process = not self.process
        raw = img.copy()
        if self.process:
            img, im_id, shape = self.process_img(img=img)
            outs = self.exe.run(self.infer_prog,
                                feed={'image': img, 'im_size': shape, 'im_id': im_id},
                                fetch_list=self.values,
                                return_numpy=False)
            res = {
                k: (np.array(v), v.recursive_sequence_lengths())
                for k, v in zip(self.keys, outs)
            }

            self.bbox_results = bbox2out(
                [res], self.clsid2catid, self.is_bbox_normalized)

        result = self.draw_bbox(raw, gray, self.catid2name,
                                self.bbox_results, self.draw_threshold)

        value = {'frame': result, 'bboxes': self.bbox_results}

        return value


