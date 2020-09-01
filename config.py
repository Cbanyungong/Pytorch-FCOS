#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================


class FCOS_R50_FPN_Multiscale_2x_Config(object):
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


        # ========= 一些设置 =========
        self.train_cfg = dict(
            lr=0.0001,   # 每隔几步保存一次模型
            batch_size=1,   # 每隔几步保存一次模型
            model_path='fcos_r50_fpn_multiscale_2x.pt',
            # model_path='./weights/step00001000.pt',

            save_iter=1000,   # 每隔几步保存一次模型
            eval_iter=5000,   # 每隔几步计算一次eval集的mAP
            max_iters=500000,   # 训练多少步
        )


        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='fcos_r50_fpn_multiscale_2x.pt',
            # model_path='./weights/step00001000.pt',
            conf_thresh=0.025,   # 验证时的分数阈值和nms_iou阈值
            nms_thresh=0.6,
            draw_image=False,    # 是否画出验证集图片
            eval_batch_size=1,   # 验证时的批大小。由于太麻烦，暂时只支持1。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='fcos_r50_fpn_multiscale_2x.pt',
            # model_path='./weights/step00001000.pt',
            conf_thresh=0.2,
            nms_thresh=0.6,
            draw_image=True,
        )


        # ============= 模型相关 =============
        self.resnet = dict(
            depth=50,
            norm_type='affine_channel',
            feature_maps=[3, 4, 5],
            use_dcn=False,
        )
        self.fpn = dict(
            num_chan=256,
        )
        self.head = dict(
            batch_size=1,
        )
        self.fcos_loss = dict(
            loss_alpha=0.25,
            loss_gamma=2.0,
            iou_loss_type='giou',  # linear_iou/giou/iou
            reg_weights=1.0,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'im_info', 'fcos_target']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=False,
        )
        # RandomFlipImage
        self.randomFlipImage = dict(
            prob=0.5,
        )
        # NormalizeImage
        self.normalizeImage = dict(
            is_channel_first=False,
            is_scale=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=[640, 672, 704, 736, 768, 800],
            max_size=1333,
            interp=1,
            use_cv2=True,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # PadBatch
        self.padBatch = dict(
            pad_to_stride=128,
            use_padded_im_info=False,
        )
        # Gt2FCOSTarget
        self.gt2FCOSTarget = dict(
            object_sizes_boundary=[64, 128, 256, 512],
            center_sampling_radius=1.5,
            downsample_ratios=[8, 16, 32, 64, 128],
            norm_reg_targets=True,
        )



class TrainConfig_2(object):
    """
    其它配置
    """
    def __init__(self):
        pass




