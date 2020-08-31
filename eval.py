#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import copy
from tools.cocotools import get_classes, catid2clsid, clsid2catid
import json
import torch

from tools.cocotools import eval
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
    # input_shape = (416, 416)
    input_shape = (608, 608)
    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.025
    nms_thresh = 0.6
    # 是否画出验证集图片
    draw_image = False
    # 验证时的批大小
    eval_batch_size = 1

    # 验证集图片的相对路径
    # eval_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'
    # anno_file = 'annotation_json/voc2012_val.json'
    eval_pre_path = '../COCO/val2017/'
    anno_file = '../COCO/annotations/instances_val2017.json'
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)

    resnet = Resnet(50)
    fpn = FPN()
    head = FCOSHead(num_classes=num_classes)
    fcos = FCOS(resnet, fpn, head)
    if use_gpu:
        fcos = fcos.cuda()
    fcos.load_state_dict(torch.load(model_path))
    fcos.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k

    _decode = Decode(conf_thresh, nms_thresh, input_shape, fcos, all_classes, use_gpu)
    box_ap = eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

