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

from config import *
from model.decode_np import Decode
from model.fcos import *
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


def multi_thread_op(i, samples, decodeImage, context, with_mixup, mixupImage,
                     photometricDistort, randomFlipImage, normalizeImage, resizeImage, permute):
    samples[i] = decodeImage(samples[i], context)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = photometricDistort(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeImage(samples[i], context)
    samples[i] = resizeImage(samples[i], context)
    samples[i] = permute(samples[i], context)

def load_weights(model, model_path):
    _state_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in _state_dict:
            shape_1 = _state_dict[k].shape
            shape_2 = pretrained_dict[k].shape
            if shape_1 == shape_2:
                new_state_dict[k] = v
            else:
                print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))
    _state_dict.update(new_state_dict)
    model.load_state_dict(_state_dict)


use_gpu = False
use_gpu = True


if __name__ == '__main__':
    cfg = FCOS_R50_FPN_Multiscale_2x_Config()
    cfg = FCOS_RT_R50_FPN_4x_Config()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

    # 步id，无需设置，会自动读。
    iter_id = 0

    # 创建模型
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)

    Fpn = select_fpn(cfg.fpn_type)
    fpn = Fpn(**cfg.fpn)

    Loss = select_loss(cfg.fcos_loss_type)
    fcos_loss = Loss(**cfg.fcos_loss)

    Head = select_head(cfg.head_type)
    head = Head(num_classes=num_classes, fcos_loss=fcos_loss, **cfg.head)

    fcos = FCOS(backbone, fpn, head)
    _decode = Decode(cfg.eval_cfg['conf_thresh'], cfg.eval_cfg['nms_thresh'], fcos, class_names, use_gpu, cfg, for_test=False)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(fcos, cfg.train_cfg['model_path'])

        strs = cfg.train_cfg['model_path'].split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
        if cfg.backbone_type == 'Resnet':
            backbone.freeze(freeze_at=5)
        # fpn.freeze()


    if use_gpu:   # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
        fcos = fcos.cuda()

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

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage()                      # mixup增强
    photometricDistort = PhotometricDistort()      # 颜色扭曲
    randomFlipImage = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
    normalizeImage = NormalizeImage(**cfg.normalizeImage)     # 先除以255归一化，再减均值除以标准差
    resizeImage = ResizeImage(**cfg.resizeImage)   # 多尺度训练，随机选一个尺度，不破坏原始宽高比地缩放。具体见代码。
    permute = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
    # batch_transforms
    padBatch = PadBatch(**cfg.padBatch)    # 由于ResizeImage()的机制特殊，这一批所有的图片的尺度不一定全相等，所以这里对齐。
    gt2FCOSTarget = Gt2FCOSTarget(**cfg.gt2FCOSTarget)   # 填写target张量。

    # 输出几个特征图
    n_features = len(cfg.gt2FCOSTarget['downsample_ratios'])

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fcos.parameters()), lr=cfg.train_cfg['lr'])   # requires_grad==True 的参数才可以被更新

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
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op, args=(i, samples, decodeImage, context, with_mixup, mixupImage,
                                                                   photometricDistort, randomFlipImage, normalizeImage, resizeImage, permute))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms
            samples = padBatch(samples, context)
            samples = gt2FCOSTarget(samples, context)

            # 整理成ndarray
            batch_images = []
            batch_labels0 = []
            batch_reg_target0 = []
            batch_centerness0 = []
            batch_labels1 = []
            batch_reg_target1 = []
            batch_centerness1 = []
            batch_labels2 = []
            batch_reg_target2 = []
            batch_centerness2 = []
            if n_features == 5:
                batch_labels3 = []
                batch_reg_target3 = []
                batch_centerness3 = []
                batch_labels4 = []
                batch_reg_target4 = []
                batch_centerness4 = []
            for sample in samples:
                im = sample['image']
                batch_images.append(np.expand_dims(im, 0))

                temp = sample['labels0']
                batch_labels0.append(np.expand_dims(temp, 0))
                temp = sample['reg_target0']
                batch_reg_target0.append(np.expand_dims(temp, 0))
                temp = sample['centerness0']
                batch_centerness0.append(np.expand_dims(temp, 0))

                temp = sample['labels1']
                batch_labels1.append(np.expand_dims(temp, 0))
                temp = sample['reg_target1']
                batch_reg_target1.append(np.expand_dims(temp, 0))
                temp = sample['centerness1']
                batch_centerness1.append(np.expand_dims(temp, 0))

                temp = sample['labels2']
                batch_labels2.append(np.expand_dims(temp, 0))
                temp = sample['reg_target2']
                batch_reg_target2.append(np.expand_dims(temp, 0))
                temp = sample['centerness2']
                batch_centerness2.append(np.expand_dims(temp, 0))

                if n_features == 5:
                    temp = sample['labels3']
                    batch_labels3.append(np.expand_dims(temp, 0))
                    temp = sample['reg_target3']
                    batch_reg_target3.append(np.expand_dims(temp, 0))
                    temp = sample['centerness3']
                    batch_centerness3.append(np.expand_dims(temp, 0))

                    temp = sample['labels4']
                    batch_labels4.append(np.expand_dims(temp, 0))
                    temp = sample['reg_target4']
                    batch_reg_target4.append(np.expand_dims(temp, 0))
                    temp = sample['centerness4']
                    batch_centerness4.append(np.expand_dims(temp, 0))
            batch_images = np.concatenate(batch_images, 0)
            batch_labels0 = np.concatenate(batch_labels0, 0)
            batch_reg_target0 = np.concatenate(batch_reg_target0, 0)
            batch_centerness0 = np.concatenate(batch_centerness0, 0)
            batch_labels1 = np.concatenate(batch_labels1, 0)
            batch_reg_target1 = np.concatenate(batch_reg_target1, 0)
            batch_centerness1 = np.concatenate(batch_centerness1, 0)
            batch_labels2 = np.concatenate(batch_labels2, 0)
            batch_reg_target2 = np.concatenate(batch_reg_target2, 0)
            batch_centerness2 = np.concatenate(batch_centerness2, 0)
            if n_features == 5:
                batch_labels3 = np.concatenate(batch_labels3, 0)
                batch_reg_target3 = np.concatenate(batch_reg_target3, 0)
                batch_centerness3 = np.concatenate(batch_centerness3, 0)
                batch_labels4 = np.concatenate(batch_labels4, 0)
                batch_reg_target4 = np.concatenate(batch_reg_target4, 0)
                batch_centerness4 = np.concatenate(batch_centerness4, 0)

            batch_images = torch.Tensor(batch_images)
            batch_labels0 = torch.Tensor(batch_labels0)
            batch_reg_target0 = torch.Tensor(batch_reg_target0)
            batch_centerness0 = torch.Tensor(batch_centerness0)
            batch_labels1 = torch.Tensor(batch_labels1)
            batch_reg_target1 = torch.Tensor(batch_reg_target1)
            batch_centerness1 = torch.Tensor(batch_centerness1)
            batch_labels2 = torch.Tensor(batch_labels2)
            batch_reg_target2 = torch.Tensor(batch_reg_target2)
            batch_centerness2 = torch.Tensor(batch_centerness2)
            if n_features == 5:
                batch_labels3 = torch.Tensor(batch_labels3)
                batch_reg_target3 = torch.Tensor(batch_reg_target3)
                batch_centerness3 = torch.Tensor(batch_centerness3)
                batch_labels4 = torch.Tensor(batch_labels4)
                batch_reg_target4 = torch.Tensor(batch_reg_target4)
                batch_centerness4 = torch.Tensor(batch_centerness4)
            if use_gpu:
                batch_images = batch_images.cuda()
                batch_labels0 = batch_labels0.cuda()
                batch_reg_target0 = batch_reg_target0.cuda()
                batch_centerness0 = batch_centerness0.cuda()
                batch_labels1 = batch_labels1.cuda()
                batch_reg_target1 = batch_reg_target1.cuda()
                batch_centerness1 = batch_centerness1.cuda()
                batch_labels2 = batch_labels2.cuda()
                batch_reg_target2 = batch_reg_target2.cuda()
                batch_centerness2 = batch_centerness2.cuda()
                if n_features == 5:
                    batch_labels3 = batch_labels3.cuda()
                    batch_reg_target3 = batch_reg_target3.cuda()
                    batch_centerness3 = batch_centerness3.cuda()
                    batch_labels4 = batch_labels4.cuda()
                    batch_reg_target4 = batch_reg_target4.cuda()
                    batch_centerness4 = batch_centerness4.cuda()
            if n_features == 3:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2]
            if n_features == 5:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2, batch_labels3, batch_labels4]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2, batch_reg_target3, batch_reg_target4]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2, batch_centerness3, batch_centerness4]
            losses = fcos(batch_images, None, eval=False, tag_labels=tag_labels, tag_bboxes=tag_bboxes, tag_centerness=tag_center)
            loss_centerness = losses['loss_centerness']
            loss_cls = losses['loss_cls']
            loss_box = losses['loss_box']
            all_loss = loss_cls + loss_box + loss_centerness

            _all_loss = all_loss.cpu().data.numpy()
            _loss_cls = loss_cls.cpu().data.numpy()
            _loss_box = loss_box.cpu().data.numpy()
            _loss_centerness = loss_centerness.cpu().data.numpy()

            # 更新权重
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            all_loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, giou_loss: {:.6f}, conf_loss: {:.6f}, cent_loss: {:.6f}, eta: {}'.format(
                    iter_id, _all_loss, _loss_box, _loss_cls, _loss_centerness, eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                save_path = './weights/step%.8d.pt' % iter_id
                torch.save(fcos.state_dict(), save_path)
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
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                fcos.eval()   # 切换到验证模式
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                fcos.train()  # 切换到训练模式

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    torch.save(fcos.state_dict(), './weights/best_model.pt')
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                exit(0)

