#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import numpy as np
import torch

from model.decode_np import Decode
from model.fcos import FCOS
from model.head import FCOSHead
from model.neck import FPN
from model.resnet import Resnet
from tools.cocotools import get_classes

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)



# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = False
use_gpu = True


if __name__ == '__main__':
    # classes_path = 'data/voc_classes.txt'
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'fcos_r50_fpn_multiscale_2x.pt'、'./weights/step00001000.pt'这些。
    model_path = 'fcos_r50_fpn_multiscale_2x.pt'
    # model_path = './weights/step00001000.pt'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.05
    conf_thresh = 0.2
    nms_thresh = 0.45

    # 是否给图片画框。不画可以提速。读图片、后处理还可以继续优化。
    draw_image = True
    # draw_image = False


    num_anchors = 3
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)

    resnet = Resnet(50)
    fpn = FPN()
    head = FCOSHead()
    fcos = FCOS(resnet, fpn, head)
    if use_gpu:
        fcos = fcos.to_cuda()
    fcos.load_state_dict(torch.load(model_path))

    # ---------------------- 加载权重 -------------------------------
    print('\nCopying...')


    def copy_conv_af(conv_unit, w, scale, offset):
        conv, af = conv_unit.conv, conv_unit.af
        conv.weight.data = torch.Tensor(w)
        af.weight.data = torch.Tensor(scale)
        af.bias.data = torch.Tensor(offset)


    def copy_conv(conv_unit, w, b):
        conv = conv_unit.conv
        conv.weight.data = torch.Tensor(w)
        conv.bias.data = torch.Tensor(b)


    def copy_conv_gn(conv_unit, w, b, scale, offset):
        conv, gn = conv_unit.conv, conv_unit.gn
        conv.weight.data = torch.Tensor(w)
        conv.bias.data = torch.Tensor(b)
        gn.weight.data = torch.Tensor(scale)
        gn.bias.data = torch.Tensor(offset)


    dic = np.load('fcos_r50_fpn_multiscale_2x.npz')

    # 获取FCOS模型的权重

    # Resnet50
    w = dic['conv1_weights']
    scale = dic['bn_conv1_scale']
    offset = dic['bn_conv1_offset']
    copy_conv_af(resnet.conv1, w, scale, offset)

    nums = [3, 4, 6, 3]
    for nid, num in enumerate(nums):
        stage_name = 'res' + str(nid + 2)
        for kk in range(num):
            block_name = stage_name + chr(ord("a") + kk)
            conv_name1 = block_name + "_branch2a"
            conv_name2 = block_name + "_branch2b"
            conv_name3 = block_name + "_branch2c"
            shortcut_name = block_name + "_branch1"

            bn_name1 = 'bn' + conv_name1[3:]
            bn_name2 = 'bn' + conv_name2[3:]
            bn_name3 = 'bn' + conv_name3[3:]
            shortcut_bn_name = 'bn' + shortcut_name[3:]

            w = dic[conv_name1 + '_weights']
            scale = dic[bn_name1 + '_scale']
            offset = dic[bn_name1 + '_offset']
            copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv1, w, scale, offset)

            w = dic[conv_name2 + '_weights']
            scale = dic[bn_name2 + '_scale']
            offset = dic[bn_name2 + '_offset']
            copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv2, w, scale, offset)

            w = dic[conv_name3 + '_weights']
            scale = dic[bn_name3 + '_scale']
            offset = dic[bn_name3 + '_offset']
            copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv3, w, scale, offset)

            # 每个stage的第一个卷积块才有4个卷积层
            if kk == 0:
                w = dic[shortcut_name + '_weights']
                scale = dic[shortcut_bn_name + '_scale']
                offset = dic[shortcut_bn_name + '_offset']
                copy_conv_af(resnet.get_block('stage%d_%d' % (2 + nid, kk)).conv4, w, scale, offset)
    # fpn, 8个卷积层
    w = dic['fpn_inner_res5_sum_w']
    b = dic['fpn_inner_res5_sum_b']
    copy_conv(fpn.s32_conv, w, b)

    w = dic['fpn_inner_res4_sum_lateral_w']
    b = dic['fpn_inner_res4_sum_lateral_b']
    copy_conv(fpn.s16_conv, w, b)

    w = dic['fpn_inner_res3_sum_lateral_w']
    b = dic['fpn_inner_res3_sum_lateral_b']
    copy_conv(fpn.s8_conv, w, b)

    w = dic['fpn_res5_sum_w']
    b = dic['fpn_res5_sum_b']
    copy_conv(fpn.sc_s32_conv, w, b)

    w = dic['fpn_res4_sum_w']
    b = dic['fpn_res4_sum_b']
    copy_conv(fpn.sc_s16_conv, w, b)

    w = dic['fpn_res3_sum_w']
    b = dic['fpn_res3_sum_b']
    copy_conv(fpn.sc_s8_conv, w, b)

    w = dic['fpn_6_w']
    b = dic['fpn_6_b']
    copy_conv(fpn.c6_conv, w, b)

    w = dic['fpn_7_w']
    b = dic['fpn_7_b']
    copy_conv(fpn.c7_conv, w, b)

    # head
    n = 5  # 有n个输出层
    num_convs = 4
    for i in range(n):  # 遍历每个输出层
        for lvl in range(0, num_convs):
            # conv + gn
            conv_cls_name = 'fcos_head_cls_tower_conv_{}'.format(lvl)
            norm_name = conv_cls_name + "_norm"
            w = dic[conv_cls_name + "_weights"]
            b = dic[conv_cls_name + "_bias"]
            scale = dic[norm_name + "_scale"]
            offset = dic[norm_name + "_offset"]
            copy_conv_gn(head.cls_convs_per_feature[i][lvl], w, b, scale, offset)

            # conv + gn
            conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
            norm_name = conv_reg_name + "_norm"
            w = dic[conv_reg_name + "_weights"]
            b = dic[conv_reg_name + "_bias"]
            scale = dic[norm_name + "_scale"]
            offset = dic[norm_name + "_offset"]
            copy_conv_gn(head.reg_convs_per_feature[i][lvl], w, b, scale, offset)

        # 类别分支最后的conv
        conv_cls_name = "fcos_head_cls"
        w = dic[conv_cls_name + "_weights"]
        b = dic[conv_cls_name + "_bias"]
        copy_conv(head.cls_convs_per_feature[i][-1], w, b)

        # 坐标分支最后的conv
        conv_reg_name = "fcos_head_reg"
        w = dic[conv_reg_name + "_weights"]
        b = dic[conv_reg_name + "_bias"]
        copy_conv(head.reg_convs_per_feature[i][-1], w, b)

        # centerness分支最后的conv
        conv_centerness_name = "fcos_head_centerness"
        w = dic[conv_centerness_name + "_weights"]
        b = dic[conv_centerness_name + "_bias"]
        copy_conv(head.ctn_convs_per_feature[i], w, b)

    # 5个scale
    fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
    i = 0
    for fpn_name in fpn_names:
        scale_i = dic["%s_scale_on_reg" % fpn_name]
        head.scales_on_reg[i].data = torch.Tensor(scale_i)
        i += 1
    if use_gpu:
        fcos = fcos.to_cuda()
    # ---------------------- 加载权重 -------------------------------


    fcos.to_eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _decode = Decode(conf_thresh, nms_thresh, input_shape, fcos, all_classes, use_gpu)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')


    path_dir = os.listdir('images/test')
    # warm up
    if use_gpu:
        for k, filename in enumerate(path_dir):
            image = cv2.imread('images/test/' + filename)
            image, boxes, scores, classes = _decode.detect_image(image, draw_image=False)
            if k == 10:
                break


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        image = cv2.imread('images/test/' + filename)
        image, boxes, scores, classes = _decode.detect_image(image, draw_image)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            cv2.imwrite('images/res/' + filename, image)
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


