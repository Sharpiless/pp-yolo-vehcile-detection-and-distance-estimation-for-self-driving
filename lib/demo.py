# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

import cv2
import scipy.misc
import tensorflow as tf
import time
import re
import argparse
import numpy as np
import imutils
from tensorflow.python.framework import graph_util
# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


parser = argparse.ArgumentParser(
    description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,
                    help='type of encoder, vgg or resnet50', default='resnet50-forward')
parser.add_argument('--checkpoint_path',  type=str,
                    help='path to a specific checkpoint to load', default='./model/model.ckpt')
parser.add_argument('--input_height',     type=int,
                    help='input height', default=256)
parser.add_argument('--input_width',      type=int,
                    help='input width', default=512)

args = parser.parse_args()
max_, min_ = None, None


def post_process_disparity(disp):
    global max_, min_
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    result = r_mask * l_disp + l_mask * \
        r_disp + (1.0 - l_mask - r_mask) * m_disp
    result = result.squeeze()
    if max_ is None:
        max_ = np.max(result)
        min_ = np.min(result)
    result = (result - min_) / (max_ - min_) * 255
    result = np.clip(result.astype(np.uint8), 0, 255)
    return result


def pre_process(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    original_height, original_width, num_channels = input_image.shape
    input_image = cv2.resize(
        input_image, (int(args.input_width), int(args.input_height)))
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)
    return input_images, original_height, original_width


def test_simple():
    """Test function."""

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # INIT
        name = 'demo'
        path = '../testttt.mp4'

        cap = cv2.VideoCapture(path)
        fps = int(cap.get(5))
        print('fps:', fps)
        t = int(1000/fps)

        size = None
        car = None

        sess.run(tf.global_variables_initializer())
        output_graph_def = tf.GraphDef()
        with open('./model.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        left = sess.graph.get_tensor_by_name("Placeholder:0")
        predict = sess.graph.get_tensor_by_name(
            "model/resnet50-forward/Conv_80/Sigmoid:0")

        y1, x1, y2, x2 = 285, 628, 709, 1710

        while True:

            _, raw = cap.read()
            if raw is None:
                break
            im = raw[y1:y2, x1:x2]

            im = imutils.resize(im, width=600)
            feed_in, original_height, original_width = pre_process(im)
            disp = sess.run(predict, feed_dict={
                left: feed_in})
            result = post_process_disparity(disp.squeeze())
            # result = cv2.applyColorMap(result, cv2.COLORMAP_JET)
            result = cv2.resize(result, (original_width, original_height))
            result = cv2.merge([result, result, result])
            # result = np.vstack([im, result])

            if size is None:
                size = (result.shape[1], result.shape[0])
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result_gray.mp4', fourcc, fps, size)
            videoWriter.write(result)
            cv2.imshow(name, result)
            cv2.imshow(name+'0', im)
            cv2.waitKey(t)

            if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
                # 点x退出
                break

        cap.release()
        videoWriter.release()
        cv2.destroyAllWindows()

    print('done!')


def main(_):

    test_simple()


if __name__ == '__main__':
    tf.app.run()
