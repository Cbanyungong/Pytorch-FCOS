#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-19 17:20:11
#   Description : pytorch_yolov4
#
# ================================================================
import torch
import numpy as np


class FCOSLoss(torch.nn.Module):
    """
    FCOSLoss
    Args:
        loss_alpha (float): alpha in focal loss
        loss_gamma (float): gamma in focal loss
        iou_loss_type(str): location loss type, IoU/GIoU/LINEAR_IoU
        reg_weights(float): weight for location loss
    """

    def __init__(self,
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 iou_loss_type="IoU",
                 reg_weights=1.0):
        super(FCOSLoss, self).__init__()
        self.loss_alpha = loss_alpha
        self.loss_gamma = loss_gamma
        self.iou_loss_type = iou_loss_type
        self.reg_weights = reg_weights

    def __flatten_tensor(self, input, channel_first=False):
        """
        Flatten a Tensor
        Args:
            input   (Variables): Input Tensor
            channel_first(bool): if true the dimension order of
                Tensor is [N, C, H, W], otherwise is [N, H, W, C]
        Return:
            input_channel_last (Variables): The flattened Tensor in channel_last style
        """
        if channel_first:
            input_channel_last = input.permute(0, 2, 3, 1)
        else:
            input_channel_last = input
        _shape = input_channel_last.shape
        N = _shape[0]
        H = _shape[1]
        W = _shape[2]
        C = _shape[3]
        input_channel_last = input_channel_last.reshape((N*H*W, C))
        return input_channel_last

    def __iou_loss(self, pred, targets, positive_mask, weights=None):
        """
        Calculate the loss for location prediction
        Args:
            pred          (Variables): bounding boxes prediction
            targets       (Variables): targets for positive samples
            positive_mask (Variables): mask of positive samples
            weights       (Variables): weights for each positive samples
        Return:
            loss (Varialbes): location loss
        """
        positive_mask = positive_mask.reshape((positive_mask.shape[0],))
        plw = pred[:, 0] * positive_mask   # [批大小*所有格子数, ]， 预测的l
        pth = pred[:, 1] * positive_mask   # [批大小*所有格子数, ]， 预测的t
        prw = pred[:, 2] * positive_mask   # [批大小*所有格子数, ]， 预测的r
        pbh = pred[:, 3] * positive_mask   # [批大小*所有格子数, ]， 预测的b
        tlw = targets[:, 0] * positive_mask   # [批大小*所有格子数, ]， 真实的l
        tth = targets[:, 1] * positive_mask   # [批大小*所有格子数, ]， 真实的t
        trw = targets[:, 2] * positive_mask   # [批大小*所有格子数, ]， 真实的r
        tbh = targets[:, 3] * positive_mask   # [批大小*所有格子数, ]， 真实的b
        area_target = (tlw + trw) * (tth + tbh)      # [批大小*所有格子数, ]， 真实的面积
        area_predict = (plw + prw) * (pth + pbh)     # [批大小*所有格子数, ]， 预测的面积
        ilw = torch.min(plw, tlw)   # [批大小*所有格子数, ]， 相交矩形的l
        irw = torch.min(prw, trw)   # [批大小*所有格子数, ]， 相交矩形的r
        ith = torch.min(pth, tth)   # [批大小*所有格子数, ]， 相交矩形的t
        ibh = torch.min(pbh, tbh)   # [批大小*所有格子数, ]， 相交矩形的b
        clw = torch.max(plw, tlw)   # [批大小*所有格子数, ]， 包围矩形的l
        crw = torch.max(prw, trw)   # [批大小*所有格子数, ]， 包围矩形的r
        cth = torch.max(pth, tth)   # [批大小*所有格子数, ]， 包围矩形的t
        cbh = torch.max(pbh, tbh)   # [批大小*所有格子数, ]， 包围矩形的b
        area_inter = (ilw + irw) * (ith + ibh)   # [批大小*所有格子数, ]， 相交矩形的面积
        ious = (area_inter + 1.0) / (
            area_predict + area_target - area_inter + 1.0)
        ious = ious * positive_mask
        if self.iou_loss_type.lower() == "linear_iou":
            loss = 1.0 - ious
        elif self.iou_loss_type.lower() == "giou":
            area_uniou = area_predict + area_target - area_inter
            area_circum = (clw + crw) * (cth + cbh) + 1e-7
            giou = ious - (area_circum - area_uniou) / area_circum
            loss = 1.0 - giou
        elif self.iou_loss_type.lower() == "iou":
            loss = 0.0 - torch.log(ious)
        else:
            raise KeyError
        loss = loss[:, np.newaxis]
        if weights is not None:
            loss = loss * weights
        return loss

    def sigmoid_focal_loss(self, x, label, fg_num, gamma=2.0, alpha=0.25):
        C = x.shape[1]
        eye = torch.eye(C + 1, device=x.device)
        one_hot = eye[label.reshape((label.shape[0],)).long()]
        pos_mask = one_hot[:, 1:]  # 正样本掩码

        p = torch.sigmoid(x)  # [批大小*所有格子数, 80]， 预测的类别概率
        pos_loss = pos_mask * (0 - torch.log(p + 1e-9)) * torch.pow(1 - p, gamma) * alpha
        neg_loss = (1.0 - pos_mask) * (0 - torch.log(1 - p + 1e-9)) * torch.pow(p, gamma) * (1 - alpha)
        focal_loss = pos_loss + neg_loss
        if fg_num > 0.5:   # 当没有gt时，即fg_num==0时，focal_loss什么都不除。
            focal_loss = focal_loss / fg_num
        return focal_loss

    def sigmoid_cross_entropy_with_logits(self, x, label):
        p = torch.sigmoid(x)
        pos_loss = label * (0 - torch.log(p + 1e-9))
        neg_loss = (1.0 - label) * (0 - torch.log(1 - p + 1e-9))
        bce_loss = pos_loss + neg_loss
        return bce_loss

    def __call__(self, cls_logits, bboxes_reg, centerness, tag_labels,
                 tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): 预测结果list。里面每个元素是[N, 80, 格子行数, 格子列数]     从 大感受野 到 小感受野
            bboxes_reg (list): 预测结果list。里面每个元素是[N,  4, 格子行数, 格子列数]     从 大感受野 到 小感受野
            centerness (list): 预测结果list。里面每个元素是[N,  1, 格子行数, 格子列数]     从 大感受野 到 小感受野
            tag_labels (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  1]     从 小感受野 到 大感受野
            tag_bboxes (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  4]     从 小感受野 到 大感受野
            tag_center (list): 真实标签list。里面每个元素是[N, 格子行数, 格子列数,  1]     从 小感受野 到 大感受野
        Return:
            loss (dict): loss composed by classification loss, bounding box
        """
        cls_logits_flatten_list = []
        bboxes_reg_flatten_list = []
        centerness_flatten_list = []
        tag_labels_flatten_list = []
        tag_bboxes_flatten_list = []
        tag_center_flatten_list = []
        num_lvl = len(cls_logits)
        for lvl in range(num_lvl):
            cls_logits_flatten_list.append(
                self.__flatten_tensor(cls_logits[num_lvl - 1 - lvl], True))   # 从 小感受野 到 大感受野 遍历cls_logits
            bboxes_reg_flatten_list.append(
                self.__flatten_tensor(bboxes_reg[num_lvl - 1 - lvl], True))
            centerness_flatten_list.append(
                self.__flatten_tensor(centerness[num_lvl - 1 - lvl], True))
            tag_labels_flatten_list.append(
                self.__flatten_tensor(tag_labels[lvl], False))   # 从 小感受野 到 大感受野 遍历tag_labels
            tag_bboxes_flatten_list.append(
                self.__flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                self.__flatten_tensor(tag_center[lvl], False))

        # 顺序都是从 小感受野 到 大感受野
        cls_logits_flatten = torch.cat(   # [批大小*所有格子数, 80]， 预测的类别
            cls_logits_flatten_list, dim=0)
        bboxes_reg_flatten = torch.cat(   # [批大小*所有格子数,  4]， 预测的lrtb
            bboxes_reg_flatten_list, dim=0)
        centerness_flatten = torch.cat(   # [批大小*所有格子数,  1]， 预测的centerness
            centerness_flatten_list, dim=0)
        tag_labels_flatten = torch.cat(   # [批大小*所有格子数,  1]， 真实的类别id
            tag_labels_flatten_list, dim=0)
        tag_bboxes_flatten = torch.cat(   # [批大小*所有格子数,  4]， 真实的lrtb
            tag_bboxes_flatten_list, dim=0)
        tag_center_flatten = torch.cat(   # [批大小*所有格子数,  1]， 真实的centerness
            tag_center_flatten_list, dim=0)

        mask_positive = tag_labels_flatten > 0        # [批大小*所有格子数,  1]， 正样本处为True
        mask_positive_float = mask_positive.float()   # [批大小*所有格子数,  1]， 正样本处为1
        num_positive_fp32 = mask_positive_float.sum()   # 这一批的正样本数
        normalize_sum = tag_center_flatten + 0
        normalize_sum = (mask_positive_float * normalize_sum).sum()   # 正样本的centerness求和

        cls_loss = self.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_fp32, gamma=self.loss_gamma, alpha=self.loss_alpha)
        reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float, tag_center_flatten) * mask_positive_float / (normalize_sum + 1e-9)
        ctn_loss = self.sigmoid_cross_entropy_with_logits(
            x=centerness_flatten,
            label=tag_center_flatten) * mask_positive_float / (num_positive_fp32 + 1e-9)
        loss_all = {
            "loss_centerness": ctn_loss.sum(),
            "loss_cls": cls_loss.sum(),
            "loss_box": reg_loss.sum()
        }
        return loss_all




