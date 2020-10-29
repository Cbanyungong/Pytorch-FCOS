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
import time
import threading
import datetime
from collections import OrderedDict
import os
import argparse
import copy

from config import *

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

parser = argparse.ArgumentParser(description='FCOS Training Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=2,
                    choices=[0, 1, 2],
                    help='0 -- fcos_r50_fpn_multiscale_2x.py;  1 -- fcos_rt_r50_fpn_4x.py;  2 -- fcos_rt_dla34_fpn_4x.py.')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


import platform
sysstr = platform.system()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False



def multi_thread_op(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms):
    for k in range(i, batch_size, num_threads):
        for sample_transform in sample_transforms:
            if isinstance(sample_transform, MixupImage):
                if with_mixup:
                    samples[k] = sample_transform(samples[k], context)
            else:
                samples[k] = sample_transform(samples[k], context)


def multi_thread_op_batch_transforms(i, num_threads, batch_size, samples, context, batch_transforms, max_shape,
                    batch_images, batch_labels0, batch_reg_target0, batch_centerness0, batch_labels1, batch_reg_target1, batch_centerness1,
                    batch_labels2, batch_reg_target2, batch_centerness2, batch_labels3, batch_reg_target3, batch_centerness3,
                    batch_labels4, batch_reg_target4, batch_centerness4, n_features):
    for k in range(i, batch_size, num_threads):
        for batch_transform in batch_transforms:
            if isinstance(batch_transform, PadBatchSingle):
                samples[k] = batch_transform(max_shape, samples[k], context)
            else:
                samples[k] = batch_transform(samples[k], context)

        # 整理成ndarray
        batch_images[k] = np.expand_dims(samples[k]['image'].astype(np.float32), 0)
        batch_labels0[k] = np.expand_dims(samples[k]['labels0'].astype(np.int32), 0)
        batch_reg_target0[k] = np.expand_dims(samples[k]['reg_target0'].astype(np.float32), 0)
        batch_centerness0[k] = np.expand_dims(samples[k]['centerness0'].astype(np.float32), 0)
        batch_labels1[k] = np.expand_dims(samples[k]['labels1'].astype(np.int32), 0)
        batch_reg_target1[k] = np.expand_dims(samples[k]['reg_target1'].astype(np.float32), 0)
        batch_centerness1[k] = np.expand_dims(samples[k]['centerness1'].astype(np.float32), 0)
        batch_labels2[k] = np.expand_dims(samples[k]['labels2'].astype(np.int32), 0)
        batch_reg_target2[k] = np.expand_dims(samples[k]['reg_target2'].astype(np.float32), 0)
        batch_centerness2[k] = np.expand_dims(samples[k]['centerness2'].astype(np.float32), 0)
        if n_features == 5:
            batch_labels3[k] = np.expand_dims(samples[k]['labels3'].astype(np.int32), 0)
            batch_reg_target3[k] = np.expand_dims(samples[k]['reg_target3'].astype(np.float32), 0)
            batch_centerness3[k] = np.expand_dims(samples[k]['centerness3'].astype(np.float32), 0)
            batch_labels4[k] = np.expand_dims(samples[k]['labels4'].astype(np.int32), 0)
            batch_reg_target4[k] = np.expand_dims(samples[k]['reg_target4'].astype(np.float32), 0)
            batch_centerness4[k] = np.expand_dims(samples[k]['centerness4'].astype(np.float32), 0)


def read_train_data(cfg,
                    train_indexes,
                    train_steps,
                    train_records,
                    batch_size,
                    _iter_id,
                    train_dic,
                    use_gpu,
                    n_features,
                    context, with_mixup, sample_transforms, batch_transforms):
    iter_id = _iter_id
    num_threads = cfg.train_cfg['num_threads']
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len >= cfg.train_cfg['max_batch']:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)

            # ==================== train ====================
            batch_images = [None] * batch_size
            batch_labels0 = [None] * batch_size
            batch_reg_target0 = [None] * batch_size
            batch_centerness0 = [None] * batch_size
            batch_labels1 = [None] * batch_size
            batch_reg_target1 = [None] * batch_size
            batch_centerness1 = [None] * batch_size
            batch_labels2 = [None] * batch_size
            batch_reg_target2 = [None] * batch_size
            batch_centerness2 = [None] * batch_size
            batch_labels3 = [None] * batch_size
            batch_reg_target3 = [None] * batch_size
            batch_centerness3 = [None] * batch_size
            batch_labels4 = [None] * batch_size
            batch_reg_target4 = [None] * batch_size
            batch_centerness4 = [None] * batch_size

            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=multi_thread_op, args=(i, num_threads, batch_size, samples, context, with_mixup, sample_transforms))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms。需要先同步PadBatch
            coarsest_stride = cfg.padBatch['pad_to_stride']
            max_shape = np.array([data['image'].shape for data in samples]).max(
                axis=0)  # max_shape=[3, max_h, max_w]
            max_shape[1] = int(  # max_h增加到最小的能被coarsest_stride=128整除的数
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(  # max_w增加到最小的能被coarsest_stride=128整除的数
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

            threads = []
            for i in range(num_threads):
                t = threading.Thread(target=multi_thread_op_batch_transforms, args=(i, num_threads, batch_size, samples, context, batch_transforms, max_shape,
                                                                   batch_images, batch_labels0, batch_reg_target0, batch_centerness0, batch_labels1, batch_reg_target1, batch_centerness1,
                                                                   batch_labels2, batch_reg_target2, batch_centerness2, batch_labels3, batch_reg_target3, batch_centerness3,
                                                                   batch_labels4, batch_reg_target4, batch_centerness4, n_features))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()


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

            dic = {}
            dic['batch_images'] = batch_images
            dic['batch_labels0'] = batch_labels0
            dic['batch_reg_target0'] = batch_reg_target0
            dic['batch_centerness0'] = batch_centerness0
            dic['batch_labels1'] = batch_labels1
            dic['batch_reg_target1'] = batch_reg_target1
            dic['batch_centerness1'] = batch_centerness1
            dic['batch_labels2'] = batch_labels2
            dic['batch_reg_target2'] = batch_reg_target2
            dic['batch_centerness2'] = batch_centerness2
            if n_features == 5:
                dic['batch_labels3'] = batch_labels3
                dic['batch_reg_target3'] = batch_reg_target3
                dic['batch_centerness3'] = batch_centerness3
                dic['batch_labels4'] = batch_labels4
                dic['batch_reg_target4'] = batch_reg_target4
                dic['batch_centerness4'] = batch_centerness4
            train_dic['%.8d'%iter_id] = dic

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                return 0



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


if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = FCOS_R50_FPN_Multiscale_2x_Config()
    elif config_file == 1:
        cfg = FCOS_RT_R50_FPN_4x_Config()
    elif config_file == 2:
        cfg = FCOS_RT_DLA34_FPN_4x_Config()

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
    head = Head(fcos_loss=fcos_loss, nms_cfg=cfg.nms_cfg, **cfg.head)
    fcos = FCOS(backbone, fpn, head)
    _decode = Decode(fcos, class_names, use_gpu, cfg, for_test=False)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(fcos, cfg.train_cfg['model_path'])

        strs = cfg.train_cfg['model_path'].split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
        backbone.freeze()

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
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

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
    padBatch = PadBatchSingle(use_padded_im_info=cfg.padBatch['use_padded_im_info'])    # 由于ResizeImage()的机制特殊，这一批所有的图片的尺度不一定全相等，所以这里对齐。
    gt2FCOSTarget = Gt2FCOSTargetSingle(**cfg.gt2FCOSTarget)   # 填写target张量。

    # 输出几个特征图
    n_features = len(cfg.gt2FCOSTarget['downsample_ratios'])

    sample_transforms = []
    sample_transforms.append(decodeImage)
    sample_transforms.append(mixupImage)
    sample_transforms.append(photometricDistort)
    sample_transforms.append(randomFlipImage)
    sample_transforms.append(normalizeImage)
    sample_transforms.append(resizeImage)
    sample_transforms.append(permute)

    batch_transforms = []
    batch_transforms.append(padBatch)
    batch_transforms.append(gt2FCOSTarget)

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fcos.parameters()), lr=cfg.train_cfg['lr'])   # requires_grad==True 的参数才可以被更新

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size

    # 读数据的线程
    train_dic ={}
    thr = threading.Thread(target=read_train_data,
                           args=(cfg,
                                 train_indexes,
                                 train_steps,
                                 train_records,
                                 batch_size,
                                 iter_id,
                                 train_dic,
                                 use_gpu,
                                 n_features,
                                 context, with_mixup, sample_transforms, batch_transforms))
    thr.start()


    best_ap_list = [0.0, 0]  #[map, iter]
    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            key_list = list(train_dic.keys())
            key_len = len(key_list)
            while key_len == 0:
                time.sleep(0.01)
                key_list = list(train_dic.keys())
                key_len = len(key_list)
            dic = train_dic.pop('%.8d'%iter_id)

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            batch_images = dic['batch_images']
            batch_labels0 = dic['batch_labels0']
            batch_reg_target0 = dic['batch_reg_target0']
            batch_centerness0 = dic['batch_centerness0']
            batch_labels1 = dic['batch_labels1']
            batch_reg_target1 = dic['batch_reg_target1']
            batch_centerness1 = dic['batch_centerness1']
            batch_labels2 = dic['batch_labels2']
            batch_reg_target2 = dic['batch_reg_target2']
            batch_centerness2 = dic['batch_centerness2']
            if n_features == 5:
                batch_labels3 = dic['batch_labels3']
                batch_reg_target3 = dic['batch_reg_target3']
                batch_centerness3 = dic['batch_centerness3']
                batch_labels4 = dic['batch_labels4']
                batch_reg_target4 = dic['batch_reg_target4']
                batch_centerness4 = dic['batch_centerness4']
            if n_features == 3:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2]
            elif n_features == 5:
                tag_labels = [batch_labels0, batch_labels1, batch_labels2, batch_labels3, batch_labels4]
                tag_bboxes = [batch_reg_target0, batch_reg_target1, batch_reg_target2, batch_reg_target3, batch_reg_target4]
                tag_center = [batch_centerness0, batch_centerness1, batch_centerness2, batch_centerness3, batch_centerness4]

            losses = fcos(batch_images, None, False, tag_labels, tag_bboxes, tag_center)
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
            if cfg.use_ema:
                fcos.update_ema_state_dict(iter_id - 1)   # 更新ema_state_dict

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, giou_loss: {:.6f}, conf_loss: {:.6f}, cent_loss: {:.6f}, eta: {}'.format(
                    iter_id, _all_loss, _loss_box, _loss_cls, _loss_centerness, eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                if cfg.use_ema:
                    fcos.apply_ema_state_dict()
                save_path = './weights/step%.8d.pt' % iter_id
                torch.save(fcos.state_dict(), save_path)
                if cfg.use_ema:
                    fcos.restore_current_state_dict()
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
                if cfg.use_ema:
                    fcos.apply_ema_state_dict()
                fcos.eval()   # 切换到验证模式
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                fcos.train()  # 切换到训练模式

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    torch.save(fcos.state_dict(), './weights/best_model.pt')
                if cfg.use_ema:
                    fcos.restore_current_state_dict()
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                exit(0)

