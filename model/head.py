#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import numpy as np
import torch
import math

from model.custom_layers import Conv2dUnit


class FCOSHead(torch.nn.Module):
    # def __init__(self,
    #              num_chan=256,
    #              add_extra_convs=False):
    #     super(FCOSHead, self).__init__()
    #     self.aaaaaaaaaaaaaaaaa = num_chan
    # def forward(self, body_feats, eval=False):
    #     pred = self.fcos_head.get_prediction(body_feats, im_info)
    #     reverse_body_feats = body_feats[::-1]   #   [s32, s16, s8]
    #     num_backbone_stages = len(reverse_body_feats)   # 3
    def __init__(self,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 num_convs=4,
                 norm_type="gn",
                 fcos_loss=None,
                 norm_reg_targets=False,
                 centerness_on_reg=False,
                 use_dcn_in_tower=False
                 # nms=MultiClassNMS(
                 #     score_threshold=0.01,
                 #     nms_top_k=1000,
                 #     keep_top_k=100,
                 #     nms_threshold=0.45,
                 #     background_label=-1).__dict__
                 ):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride[::-1]
        self.prior_prob = prior_prob
        self.num_convs = num_convs
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg
        self.use_dcn_in_tower = use_dcn_in_tower
        self.norm_type = norm_type
        self.fcos_loss = fcos_loss
        self.batch_size = 8
        # self.nms = nms
        # if isinstance(nms, dict):
        #     self.nms = MultiClassNMS(**nms)


        self.scales_on_reg = []       # 回归分支（预测框坐标）的系数
        self.cls_convs_per_feature = []   # 每个fpn输出特征图再进行卷积的卷积层，用于预测类别
        self.reg_convs_per_feature = []   # 每个fpn输出特征图再进行卷积的卷积层，用于预测坐标
        self.ctn_convs_per_feature = []   # 用于预测centerness
        n = len(self.fpn_stride)      # 有n个输出层
        for i in range(n):     # 遍历每个输出层
            scale = torch.nn.Parameter(torch.ones(1, ))
            self.scales_on_reg.append(scale)
            cls_convs_this_feature = []   # 这个fpn输出特征图再进行卷积的卷积层，用于预测类别
            reg_convs_this_feature = []   # 这个fpn输出特征图再进行卷积的卷积层，用于预测坐标
            for lvl in range(0, self.num_convs):
                # 使用gn，组数是32，而且带激活relu
                cls_conv_layer = Conv2dUnit(256, 256, 3, stride=1, bias_attr=True, gn=1, groups=32, act='relu')
                cls_convs_this_feature.append(cls_conv_layer)
                reg_conv_layer = Conv2dUnit(256, 256, 3, stride=1, bias_attr=True, gn=1, groups=32, act='relu')
                reg_convs_this_feature.append(reg_conv_layer)

            # 类别分支最后的卷积
            cls_last_conv_layer = Conv2dUnit(256, self.num_classes, 3, stride=1, bias_attr=True, act=None)
            # 坐标分支最后的卷积
            reg_last_conv_layer = Conv2dUnit(256, 4, 3, stride=1, bias_attr=True, act=None)
            # centerness分支
            ctn_last_conv_layer = Conv2dUnit(256, 1, 3, stride=1, bias_attr=True, act=None)

            cls_convs_this_feature.append(cls_last_conv_layer)
            reg_convs_this_feature.append(reg_last_conv_layer)
            self.cls_convs_per_feature.append(cls_convs_this_feature)
            self.reg_convs_per_feature.append(reg_convs_this_feature)
            self.ctn_convs_per_feature.append(ctn_last_conv_layer)

        self.relu = torch.nn.ReLU()


    def _fcos_head(self, features, fpn_stride, i, is_training=False):
        """
        Args:
            features (Variables): feature map from FPN
            fpn_stride     (int): the stride of current feature map
            is_training   (bool): whether is train or test mode
        """
        fpn_scale = self.scales_on_reg[i]
        subnet_blob_cls = features
        subnet_blob_reg = features
        # if self.use_dcn_in_tower:
        #     conv_norm = DeformConvNorm
        # else:
        #     conv_norm = ConvNorm
        for lvl in range(0, self.num_convs):
            subnet_blob_cls = self.cls_convs_per_feature[i][lvl](subnet_blob_cls)
            subnet_blob_reg = self.reg_convs_per_feature[i][lvl](subnet_blob_reg)


        cls_logits = self.cls_convs_per_feature[i][-1](subnet_blob_cls)   # 通道数变成类别数
        bbox_reg = self.reg_convs_per_feature[i][-1](subnet_blob_reg)     # 通道数变成4
        bbox_reg = bbox_reg * fpn_scale     # 预测坐标的特征图整体乘上fpn_scale，是一个可学习参数
        # 如果 归一化坐标分支，bbox_reg进行relu激活
        if self.norm_reg_targets:
            bbox_reg = self.relu(bbox_reg)
            if not is_training:   # 验证状态的话，bbox_reg再乘以下采样倍率
                bbox_reg = bbox_reg * fpn_stride
        else:
            bbox_reg = torch.exp(bbox_reg)


        # ============= centerness分支，默认是用坐标分支接4个卷积层之后的结果subnet_blob_reg =============
        if self.centerness_on_reg:
            centerness = self.ctn_convs_per_feature[i](subnet_blob_reg)
        else:
            centerness = self.ctn_convs_per_feature[i](subnet_blob_cls)
        return cls_logits, bbox_reg, centerness

    def _get_output(self, body_feats, is_training=False):
        """
        Args:
            body_feates (list): the list of fpn feature maps
            is_training (bool): whether is train or test mode
        Return:
            cls_logits (Variables): prediction for classification
            bboxes_reg (Variables): prediction for bounding box
            centerness (Variables): prediction for ceterness
        """
        cls_logits = []
        bboxes_reg = []
        centerness = []
        assert len(body_feats) == len(self.fpn_stride), \
            "The size of body_feats is not equal to size of fpn_stride"
        i = 0
        for features, fpn_stride in zip(body_feats, self.fpn_stride):
            cls_pred, bbox_pred, ctn_pred = self._fcos_head(
                features, fpn_stride, i, is_training=is_training)
            cls_logits.append(cls_pred)
            bboxes_reg.append(bbox_pred)
            centerness.append(ctn_pred)
            i += 1
        return cls_logits, bboxes_reg, centerness

    def _compute_locations(self, features):
        """
        Args:
            features (list): List of Variables for FPN feature maps
        Return:
            Anchor points for each feature map pixel
        """
        locations = []
        for lvl, feature in enumerate(features):
            shape_fm = feature.size()
            h = shape_fm[2]
            w = shape_fm[3]
            fpn_stride = self.fpn_stride[lvl]
            shift_x = torch.arange(0, w, dtype=torch.float32) * fpn_stride   # 生成x偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
            shift_y = torch.arange(0, h, dtype=torch.float32) * fpn_stride   # 生成y偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
            shift_x = shift_x.unsqueeze(0)   # [1, w]
            shift_y = shift_y.unsqueeze(1)   # [h, 1]
            shift_x = shift_x.repeat((h, 1))   # [h, w]
            shift_y = shift_y.repeat((1, w))   # [h, w]
            shift_x = shift_x.reshape((h * w, 1))   # [h*w, 1]
            shift_y = shift_y.reshape((h * w, 1))   # [h*w, 1]
            location = torch.cat([shift_x, shift_y], dim=-1)   # [h*w, 2]  格子左上角的坐标，单位是1像素。顺序是先第一行格子从左到右，再到第二行格子从左到右，...
            location += fpn_stride // 2                        # [h*w, 2]  格子中心点的坐标，单位是1像素。顺序是先第一行格子从左到右，再到第二行格子从左到右，...
            locations.append(location)
        return locations

    def _postprocessing_by_level(self, locations, box_cls, box_reg, box_ctn,
                                 im_info):
        """
        Args:
            locations (Variables): anchor points for current layer
            box_cls   (Variables): categories prediction
            box_reg   (Variables): bounding box prediction
            box_ctn   (Variables): centerness prediction
            im_info   (Variables): [h, w, scale] for input images
        Return:
            box_cls_ch_last  (Variables): score for each category, in [N, C, M]
                C is the number of classes and M is the number of anchor points
            box_reg_decoding (Variables): decoded bounding box, in [N, M, 4]
                last dimension is [x1, y1, x2, y2]
        """
        batch_size = self.batch_size
        num_classes = self.num_classes

        # =========== 类别概率，[N, 80, H*W] ===========
        box_cls_ch_last = box_cls.reshape(
            (batch_size, num_classes, box_cls.size()[2] * box_cls.size()[3]))  # [N, 80, H*W]
        box_cls_ch_last = torch.sigmoid(box_cls_ch_last)  # 类别概率用sigmoid()激活，[N, 80, H*W]

        # =========== 坐标(4个偏移)，[N, H*W, 4] ===========
        box_reg_ch_last = box_reg.permute(0, 2, 3, 1)  # [N, H, W, 4]
        box_reg_ch_last = box_reg_ch_last.reshape(
            (batch_size, box_reg_ch_last.size()[1] * box_reg_ch_last.size()[2], 4))  # [N, H*W, 4]，坐标不用再接激活层，直接预测。

        # =========== centerness，[N, 1, H*W] ===========
        box_ctn_ch_last = box_ctn.reshape((batch_size, 1, box_ctn.size()[2] * box_ctn.size()[3]))  # [N, 1, H*W]
        box_ctn_ch_last = torch.sigmoid(box_ctn_ch_last)  # centerness用sigmoid()激活，[N, 1, H*W]

        box_reg_decoding = torch.cat(  # [N, H*W, 4]
            [
                locations[:, 0:1] - box_reg_ch_last[:, :, 0:1],  # 左上角x坐标
                locations[:, 1:2] - box_reg_ch_last[:, :, 1:2],  # 左上角y坐标
                locations[:, 0:1] + box_reg_ch_last[:, :, 2:3],  # 右下角x坐标
                locations[:, 1:2] + box_reg_ch_last[:, :, 3:4]  # 右下角y坐标
            ],
            dim=-1)
        # # recover the location to original image
        im_scale = im_info[:, 2]  # [N, ]
        im_scale = im_scale[:, np.newaxis, np.newaxis]  # [N, 1, 1]
        box_reg_decoding = box_reg_decoding / im_scale  # [N, H*W, 4]，最终坐标=坐标*图片缩放因子
        box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last  # [N, 80, H*W]，最终分数=类别概率*centerness
        return box_cls_ch_last, box_reg_decoding

    def _post_processing(self, locations, cls_logits, bboxes_reg, centerness,
                         im_info):
        """
        Args:
            locations   (list): List of Variables composed by center of each anchor point
            cls_logits  (list): List of Variables for class prediction
            bboxes_reg  (list): List of Variables for bounding box prediction
            centerness  (list): List of Variables for centerness prediction
            im_info(Variables): [h, w, scale] for input images
        Return:
            pred (LoDTensor): predicted bounding box after nms,
                the shape is n x 6, last dimension is [label, score, xmin, ymin, xmax, ymax]
        """
        pred_boxes_ = []
        pred_scores_ = []
        for _, (
                pts, cls, box, ctn
        ) in enumerate(zip(locations, cls_logits, bboxes_reg, centerness)):
            pred_scores_lvl, pred_boxes_lvl = self._postprocessing_by_level(
                pts, cls, box, ctn, im_info)
            pred_boxes_.append(pred_boxes_lvl)
            pred_scores_.append(pred_scores_lvl)
        pred_boxes = torch.cat(pred_boxes_, dim=1)
        pred_scores = torch.cat(pred_scores_, dim=2)
        # pred = self.nms(pred_boxes, pred_scores)
        # return pred
        return pred_boxes, pred_scores

    def get_loss(self, input, tag_labels, tag_bboxes, tag_centerness):
        """
        Calculate the loss for FCOS
        Args:
            input           (list): List of Variables for feature maps from FPN layers
            tag_labels     (Variables): category targets for each anchor point
            tag_bboxes     (Variables): bounding boxes  targets for positive samples
            tag_centerness (Variables): centerness targets for positive samples
        Return:
            loss (dict): loss composed by classification loss, bounding box
                regression loss and centerness regression loss
        """
        cls_logits, bboxes_reg, centerness = self._get_output(
            input, is_training=True)
        loss = self.fcos_loss(cls_logits, bboxes_reg, centerness, tag_labels,
                              tag_bboxes, tag_centerness)
        return loss

    def get_prediction(self, input, im_info):
        """
        Decode the prediction
        Args:
            input: [c7, c6, c5, c4, c3]
            im_info(Variables): [h, w, scale] for input images
        Return:
            the bounding box prediction
        """
        # cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
        # bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
        # centerness里面每个元素是[N,  1, 格子行数, 格子列数]
        cls_logits, bboxes_reg, centerness = self._get_output(
            input, is_training=False)
        locations = self._compute_locations(input)
        pred = self._post_processing(locations, cls_logits, bboxes_reg,
                                     centerness, im_info)
        # return {"bbox": pred}
        return pred


