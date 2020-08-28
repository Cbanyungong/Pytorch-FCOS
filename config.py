#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================


class TrainConfig(object):
    """
    train.py里需要的配置
    """
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径

        # COCO数据集
        self.train_path = '../COCO/annotations/instances_train2017.json'
        self.val_path = '../COCO/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../COCO/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '../COCO/val2017/'  # 验证集图片相对路径

        # 训练时若预测框与所有的gt小于阈值self.iou_loss_thresh时视为反例
        self.iou_loss_thresh = 0.7

        # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
        self.pattern = 1
        self.lr = 0.0001
        self.batch_size = 1
        # 如果self.pattern = 1，需要指定self.model_path表示从哪个模型读取权重继续训练。
        self.model_path = 'fcos_r50_fpn_multiscale_2x.pt'
        # self.model_path = './weights/step00001000.pt'

        # ========= 一些设置 =========
        # 每隔几步保存一次模型
        self.save_iter = 1000
        # 每隔几步计算一次eval集的mAP
        self.eval_iter = 5000
        # 训练多少步
        self.max_iters = 800000


        # 验证
        # self.input_shape越大，精度会上升，但速度会下降。
        # self.input_shape = (320, 320)
        # self.input_shape = (416, 416)
        self.input_shape = (608, 608)
        # 验证时的分数阈值和nms_iou阈值
        self.conf_thresh = 0.001
        self.nms_thresh = 0.45
        # 是否画出验证集图片
        self.draw_image = False
        # 验证时的批大小
        self.eval_batch_size = 4


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'im_info', 'fcos_target']}
        # DecodeImage
        self.di_to_rgb = True
        self.with_mixup = False
        # RandomFlipImage
        self.rfi_prob = 0.5
        # NormalizeImage
        self.is_channel_first = False
        self.is_scale = True
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # ResizeImage
        self.target_size = [640, 672, 704, 736, 768, 800]
        self.max_size = 1333
        self.interp = 1
        self.use_cv2 = True
        # Permute
        self.p_to_rgb = False
        self.channel_first = True
        # PadBatch
        self.pad_to_stride = 128
        self.use_padded_im_info = False
        # Gt2FCOSTarget
        self.object_sizes_boundary = [64, 128, 256, 512]
        self.center_sampling_radius = 1.5
        self.downsample_ratios = [8, 16, 32, 64, 128]
        self.norm_reg_targets = True



class TrainConfig_2(object):
    """
    其它配置
    """
    def __init__(self):
        pass




