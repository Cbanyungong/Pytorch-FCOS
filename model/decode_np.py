# -*- coding: utf-8 -*-
import torch
import random
import colorsys
import cv2
import threading
import os
import numpy as np


class Decode(object):
    def __init__(self, obj_threshold, nms_threshold, _fcos, all_classes, use_gpu):
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self.all_classes = all_classes
        self.num_classes = len(self.all_classes)
        self._fcos = _fcos
        self.use_gpu = use_gpu

    # 处理一张图片
    def detect_image(self, image, draw_image):
        pimage, im_info = self.process_image(np.copy(image))

        boxes, scores, classes = self.predict(pimage, im_info, image.shape)
        if boxes is not None and draw_image:
            self.draw(image, boxes, scores, classes)
        return image, boxes, scores, classes

    # 处理一批图片
    def detect_batch(self, batch_img, draw_image):
        batch_size = len(batch_img)
        result_image, result_boxes, result_scores, result_classes = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
        batch = []
        batch_im_info = []

        for image in batch_img:
            pimage, im_info = self.process_image(np.copy(image))
            batch.append(pimage)
            batch_im_info.append(im_info)
        batch = np.concatenate(batch, axis=0)
        batch_im_info = np.concatenate(batch_im_info, axis=0)
        batch = torch.Tensor(batch)
        batch_im_info = torch.Tensor(batch_im_info)
        if self.use_gpu:
            batch = batch.cuda()
            batch_im_info = batch_im_info.cuda()
        pred_boxes, pred_scores = self._fcos(batch, batch_im_info, eval=True)
        pred_boxes = pred_boxes.cpu().detach().numpy()    # [N, 所有格子, 4]，最终坐标
        pred_scores = pred_scores.cpu().detach().numpy()  # [N, 80, 所有格子]，最终分数

        boxes, scores, classes = self._fcos_out(pred_boxes[0], pred_scores[0])
        if boxes is not None and draw_image:
            self.draw(batch_img[0], boxes, scores, classes)
        result_image[0] = batch_img[0]
        result_boxes[0] = boxes
        result_scores[0] = scores
        result_classes[0] = classes
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # NormalizeImage
        mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :].astype(np.float32)
        std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :].astype(np.float32)
        pimage = img.astype(np.float32) / 255.
        pimage -= mean
        im = pimage / std

        # ResizeImage。影响FCOS速度的主要原因是图片分辨率过大。
        max_size = 1333
        target_size = 800
        target_size = 320
        max_size = target_size * (1333.0/800.0)

        use_cv2 = True
        interp = 1
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        if isinstance(target_size, list):
            # Case for multi-scale training
            selected_size = random.choice(target_size)
        else:
            selected_size = target_size
        if max_size != 0:
            im_scale = float(selected_size) / float(im_size_min)
            # Prevent the biggest axis from being more than max_size
            if np.round(im_scale * im_size_max) > max_size:
                im_scale = float(max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale

            resize_w = im_scale_x * float(im_shape[1])
            resize_h = im_scale_y * float(im_shape[0])
            im_info = [resize_h, resize_w, im_scale]
            # if 'im_info' in sample and sample['im_info'][2] != 1.:
            #     sample['im_info'] = np.append(
            #         list(sample['im_info']), im_info).astype(np.float32)
            # else:
            #     sample['im_info'] = np.array(im_info).astype(np.float32)
        else:
            im_scale_x = float(selected_size) / float(im_shape[1])
            im_scale_y = float(selected_size) / float(im_shape[0])

            resize_w = selected_size
            resize_h = selected_size

        if use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=interp)

        # Permute
        im = im.transpose(2, 0, 1)

        # PadBatch
        use_padded_im_info = True
        coarsest_stride = 128
        max_shape = np.array([im.shape]).max(axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)   # np.ceil()上取整
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)   # np.ceil()上取整

        im_c, im_h, im_w = im.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        if use_padded_im_info:
            im_info[:2] = max_shape[1:3]

        pimage = np.expand_dims(padding_im, axis=0)
        im_info = np.expand_dims(im_info, axis=0)
        return pimage, im_info

    def predict(self, image, im_info, shape):
        image = torch.Tensor(image)
        im_info = torch.Tensor(im_info)
        if self.use_gpu:
            image = image.cuda()
            im_info = im_info.cuda()
        pred_boxes, pred_scores = self._fcos(image, im_info, eval=True)
        pred_boxes = pred_boxes.cpu().detach().numpy()    # [N, 所有格子, 4]，最终坐标
        pred_scores = pred_scores.cpu().detach().numpy()  # [N, 80, 所有格子]，最终分数

        # numpy后处理
        boxes, scores, classes = self._fcos_out(pred_boxes[0], pred_scores[0])

        return boxes, scores, classes

    def _nms_boxes(self, boxes, scores):
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - x
        h = boxes[:, 3] - y

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self._t2)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        return keep


    def _fcos_out(self, pred_boxes, pred_scores):
        '''
        :param pred_boxes:   [所有格子, 4]，最终坐标
        :param pred_scores:  [80, 所有格子]，最终分数
        :return:
        '''
        # 分数过滤
        box_classes = np.argmax(pred_scores, axis=0)
        box_class_scores = np.max(pred_scores, axis=0)
        pos = np.where(box_class_scores >= self._t1)

        boxes = pred_boxes[pos]         # [M, 4]
        classes = box_classes[pos]      # [M, ]
        scores = box_class_scores[pos]  # [M, ]


        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            keep = self._nms_boxes(b, s)

            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, scores, classes


