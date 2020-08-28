#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import cv2
from collections import deque
import math
import json
import time
import threading
import datetime
import random
import copy
import numpy as np
from collections import OrderedDict
import os
import torch

from config import TrainConfig
from model.losses import FCOSLoss
from model.decode_np import Decode
from model.fcos import FCOS
from model.head import FCOSHead
from model.neck import FPN
from model.resnet import Resnet
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


import platform
sysstr = platform.system()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False


def multi_thread_op(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                     photometricDistort, randomCrop, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH):
    samples[i] = decodeImage(samples[i], context, train_dataset)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = photometricDistort(samples[i], context)
    samples[i] = randomCrop(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeBox(samples[i], context)
    samples[i] = padBox(samples[i], context)
    samples[i] = bboxXYXY2XYWH(samples[i], context)


use_gpu = False
use_gpu = True


if __name__ == '__main__':
    cfg = TrainConfig()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

    # 步id，无需设置，会自动读。
    iter_id = 0

    # 创建模型
    resnet = Resnet(50)
    fpn = FPN()
    head = FCOSHead()
    fcos = FCOS(resnet, fpn, head)
    _decode = Decode(cfg.conf_thresh, cfg.nms_thresh, cfg.input_shape, fcos, class_names, use_gpu)

    # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
    pattern = cfg.pattern
    if pattern == 1:
        # 加载参数, 跳过形状不匹配的。
        yolo_state_dict = yolo.state_dict()
        pretrained_dict = torch.load(cfg.model_path)
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if k in yolo_state_dict:
                shape_1 = yolo_state_dict[k].shape
                shape_2 = pretrained_dict[k].shape
                if shape_1 == shape_2:
                    new_state_dict[k] = v
                else:
                    print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))
        yolo_state_dict.update(new_state_dict)
        yolo.load_state_dict(yolo_state_dict)


        strs = cfg.model_path.split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。6G的卡建议这样配置。11G的卡建议不冻结。
        freeze_before = 'conv086'
        for param in yolo.named_parameters():
            if freeze_before in param[0]:
                break
            else:
                print('freeze %s' % param[0])
                param[1].requires_grad = False
    elif pattern == 0:
        pass


    # 建立损失函数
    fcos_loss = FCOSLoss(num_classes, cfg.iou_loss_thresh, _anchors)
    if use_gpu:   # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
        fcos = fcos.cuda()
        fcos_loss = fcos_loss.cuda()

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            val_images = dataset['images']

    batch_size = cfg.batch_size
    with_mixup = cfg.with_mixup
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(with_mixup=with_mixup)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage()                   # mixup增强
    photometricDistort = PhotometricDistort()   # 颜色扭曲
    randomCrop = RandomCrop()                   # 随机裁剪
    randomFlipImage = RandomFlipImage()         # 随机翻转
    normalizeBox = NormalizeBox()               # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
    padBox = PadBox(cfg.num_max_boxes)          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
    bboxXYXY2XYWH = BboxXYXY2XYWH()             # sample['gt_bbox']被改写为cx_cy_w_h格式。
    # batch_transforms
    randomShape = RandomShape()                 # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
    normalizeImage = NormalizeImage(is_scale=True, is_channel_first=False)  # 图片归一化。直接除以255。
    gt2YoloTarget = Gt2YoloTarget(cfg.anchors,
                                  cfg.anchor_masks,
                                  cfg.downsample_ratios,
                                  num_classes)             # 填写target0、target1、target2张量。

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, yolo.parameters()), lr=cfg.lr)   # requires_grad==True 的参数才可以被更新

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  #[map, iter]
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.max_iters - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op, args=(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                                                                   photometricDistort, randomCrop, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms
            samples = randomShape(samples, context)
            samples = normalizeImage(samples, context)
            batch_image, batch_label, batch_gt_bbox = gt2YoloTarget(samples, context)

            # 一些变换
            batch_image = batch_image.transpose(0, 3, 1, 2)
            batch_image = torch.Tensor(batch_image)

            batch_label[2] = torch.Tensor(batch_label[2])
            batch_label[1] = torch.Tensor(batch_label[1])
            batch_label[0] = torch.Tensor(batch_label[0])

            batch_gt_bbox = torch.Tensor(batch_gt_bbox)

            if use_cuda:
                batch_image = batch_image.cuda()
                batch_label[2] = batch_label[2].cuda()
                batch_label[1] = batch_label[1].cuda()
                batch_label[0] = batch_label[0].cuda()
                batch_gt_bbox = batch_gt_bbox.cuda()

            l_pred, m_pred, s_pred = yolo(batch_image)  # 直接卷积后的输出
            args = [l_pred, m_pred, s_pred, batch_label[2], batch_label[1], batch_label[0], batch_gt_bbox]
            losses = yolo_loss(args)
            if use_cuda:
                all_loss = losses[0].cpu().data.numpy()
                ciou_loss = losses[1].cpu().data.numpy()
                conf_loss = losses[2].cpu().data.numpy()
                prob_loss = losses[3].cpu().data.numpy()
            else:
                all_loss = losses[0].data.numpy()
                ciou_loss = losses[1].data.numpy()
                conf_loss = losses[2].data.numpy()
                prob_loss = losses[3].data.numpy()
            # 更新权重
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            losses[0].backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, ciou_loss: {:.6f}, conf_loss: {:.6f}, prob_loss: {:.6f}, eta: {}'.format(
                    iter_id, all_loss, ciou_loss, conf_loss, prob_loss, eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.save_iter == 0:
                save_path = './weights/step%.8d.pt' % iter_id
                torch.save(yolo.state_dict(), save_path)
                path_dir = os.listdir('./weights')
                steps = []
                names = []
                for name in path_dir:
                    if name[len(name) - 2:len(name)] == 'pt' and name[0:4] == 'step':
                        step = int(name[4:12])
                        steps.append(step)
                        names.append(name)
                if len(steps) > 10:
                    i = steps.index(min(steps))
                    os.remove('./weights/'+names[i])
                logger.info('Save model to {}'.format(save_path))

            # ==================== eval ====================
            if iter_id % cfg.eval_iter == 0:
                yolo.eval()   # 切换到验证模式
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_batch_size, _clsid2catid, cfg.draw_image)
                logger.info("box ap: %.3f" % (box_ap[0], ))
                yolo.train()  # 切换到训练模式

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    torch.save(yolo.state_dict(), './weights/best_model.pt')
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.max_iters:
                logger.info('Done.')
                exit(0)

