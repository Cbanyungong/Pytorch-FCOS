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

from config import *
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
    cfg = FCOS_R50_FPN_Multiscale_2x_Config()
    cfg = FCOS_RT_R50_FPN_4x_Config()


    # 读取的模型
    model_path = cfg.eval_cfg['model_path']

    # 分数阈值和nms_iou阈值
    conf_thresh = cfg.eval_cfg['conf_thresh']
    nms_thresh = cfg.eval_cfg['nms_thresh']

    # 是否给图片画框。
    draw_image = cfg.eval_cfg['draw_image']

    # 验证时的批大小
    eval_batch_size = cfg.eval_cfg['eval_batch_size']

    # 验证集图片的相对路径
    eval_pre_path = cfg.val_pre_path
    anno_file = cfg.val_path
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    all_classes = get_classes(cfg.classes_path)
    num_classes = len(all_classes)


    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    Fpn = select_fpn(cfg.fpn_type)
    fpn = Fpn(**cfg.fpn)
    Head = select_head(cfg.head_type)
    head = Head(num_classes=num_classes, **cfg.head)
    fcos = FCOS(backbone, fpn, head)
    if use_gpu:
        fcos = fcos.cuda()
    fcos.load_state_dict(torch.load(model_path))
    fcos.eval()  # 必须调用model.eval()来设置dropout和batch normalization layers在运行推理前，切换到评估模式. 不这样做的化会产生不一致的推理结果.

    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k

    _decode = Decode(conf_thresh, nms_thresh, fcos, all_classes, use_gpu, cfg, for_test=False)
    box_ap = eval(_decode, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

