#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import random
import colorsys
import cv2
import threading
import os
import torch
import numpy as np

from tools.transform import *


class Decode(object):
    def __init__(self, _model, all_classes, use_gpu, cfg, for_test=True):
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self._model = _model
        self.use_gpu = use_gpu

        # 图片预处理
        self.context = cfg.context
        # sample_transforms
        self.to_rgb = cfg.decodeImage['to_rgb']
        self.normalizeImage = NormalizeImage(**cfg.normalizeImage)  # 先除以255归一化，再减均值除以标准差
        target_size = cfg.eval_cfg['target_size']
        max_size = cfg.eval_cfg['max_size']
        if for_test:
            target_size = cfg.test_cfg['target_size']
            max_size = cfg.test_cfg['max_size']
        self.resizeImage = ResizeImage(target_size=target_size,
                                       max_size=max_size,
                                       interp=cfg.resizeImage['interp'],
                                       use_cv2=cfg.resizeImage['use_cv2'])  # 多尺度训练，随机选一个尺度，不破坏原始宽高比地缩放。具体见代码。
        self.permute = Permute(**cfg.permute)  # 图片从HWC格式变成CHW格式
        # batch_transforms
        self.padBatch = PadBatch(**cfg.padBatch)  # 由于ResizeImage()的机制特殊，这一批所有的图片的尺度不一定全相等，所以这里对齐。


    # 处理一张图片
    def detect_image(self, image, pimage, im_size, draw_image, draw_thresh=0.0):
        pred = self.predict(pimage, im_size)   # [bs, M, 6]
        if pred[0][0][0] < 0.0:
            boxes = np.array([])
            classes = np.array([])
            scores = np.array([])
        else:
            boxes = pred[0, :, 2:]
            scores = pred[0, :, 1]
            classes = pred[0, :, 0].astype(np.int32)
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]         # [M, 4]
            scores2 = scores[pos]       # [M, ]
            classes2 = classes[pos]     # [M, ]
            self.draw(image, boxes2, scores2, classes2)
        return image, boxes, scores, classes

    # 多线程后处理
    def multi_thread_post(self, i, pred, result_image, result_boxes, result_scores, result_classes, batch_img, draw_image, draw_thresh):
        if pred[i][0][0] < 0.0:
            boxes = np.array([])
            classes = np.array([])
            scores = np.array([])
        else:
            boxes = pred[i, :, 2:]
            scores = pred[i, :, 1]
            classes = pred[i, :, 0].astype(np.int32)
            pos = np.where(scores >= 0.0)
            boxes = boxes[pos]  # [M, 4]
            scores = scores[pos]  # [M, ]
            classes = classes[pos]  # [M, ]
        if len(scores) > 0 and draw_image:
            pos = np.where(scores >= draw_thresh)
            boxes2 = boxes[pos]  # [M, 4]
            scores2 = scores[pos]  # [M, ]
            classes2 = classes[pos]  # [M, ]
            self.draw(batch_img[i], boxes2, scores2, classes2)
        result_image[i] = batch_img[i]
        result_boxes[i] = boxes
        result_scores[i] = scores
        result_classes[i] = classes

    # 处理一批图片
    def detect_batch(self, batch_img, batch_pimage, batch_im_size, draw_image, draw_thresh=0.0):
        batch_size = len(batch_img)
        result_image, result_boxes, result_scores, result_classes = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size

        pred = self.predict(batch_pimage, batch_im_size)   # [bs, M, 6]

        threads = []
        for i in range(batch_size):
            t = threading.Thread(target=self.multi_thread_post,
                                 args=(i, pred, result_image, result_boxes, result_scores, result_classes, batch_img, draw_image, draw_thresh))
            threads.append(t)
            t.start()
        # 等待所有线程任务结束。
        for t in threads:
            t.join()
        return result_image, result_boxes, result_scores, result_classes

    def draw(self, image, boxes, scores, classes):
        image_h, image_w, _ = image.shape
        # 定义颜色
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for box, score, cl in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            left = max(0, np.floor(x0 + 0.5).astype(int))
            top = max(0, np.floor(y0 + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
            bbox_color = colors[cl]
            # bbox_thick = 1 if min(image_h, image_w) < 400 else 2
            bbox_thick = 1
            cv2.rectangle(image, (left, top), (right, bottom), bbox_color, bbox_thick)
            bbox_mess = '%s: %.2f' % (self.all_classes[cl], score)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, (left, top), (left + t_size[0], top - t_size[1] - 3), bbox_color, -1)
            cv2.putText(image, bbox_mess, (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    def process_image(self, img):
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        context = self.context
        sample = {}
        sample['image'] = img
        sample['h'] = img.shape[0]
        sample['w'] = img.shape[1]

        sample = self.normalizeImage(sample, context)
        sample = self.resizeImage(sample, context)
        sample = self.permute(sample, context)
        samples = self.padBatch([sample], context)

        pimage = np.expand_dims(samples[0]['image'], axis=0)
        im_info = np.expand_dims(samples[0]['im_info'], axis=0)
        return pimage, im_info

    def predict(self, image, im_size):
        image = torch.Tensor(image)
        im_size = torch.Tensor(im_size)
        if self.use_gpu:
            image = image.cuda()
            im_size = im_size.cuda()
        pred = self._model(image, im_size)
        pred = pred.cpu().detach().numpy()   # [bs, M, 6]
        return pred



