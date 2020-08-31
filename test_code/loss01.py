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
import paddle.fluid as fluid
import paddle.fluid.layers as P
import torch



class FCOSLossPaddle(object):
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
            input_channel_last = fluid.layers.transpose(
                input, perm=[0, 2, 3, 1])
        else:
            input_channel_last = input
        input_channel_last = fluid.layers.flatten(input_channel_last, axis=3)
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
        plw = pred[:, 0] * positive_mask
        pth = pred[:, 1] * positive_mask
        prw = pred[:, 2] * positive_mask
        pbh = pred[:, 3] * positive_mask
        tlw = targets[:, 0] * positive_mask
        tth = targets[:, 1] * positive_mask
        trw = targets[:, 2] * positive_mask
        tbh = targets[:, 3] * positive_mask
        tlw.stop_gradient = True
        trw.stop_gradient = True
        tth.stop_gradient = True
        tbh.stop_gradient = True
        area_target = (tlw + trw) * (tth + tbh)
        area_predict = (plw + prw) * (pth + pbh)
        ilw = fluid.layers.elementwise_min(plw, tlw)
        irw = fluid.layers.elementwise_min(prw, trw)
        ith = fluid.layers.elementwise_min(pth, tth)
        ibh = fluid.layers.elementwise_min(pbh, tbh)
        clw = fluid.layers.elementwise_max(plw, tlw)
        crw = fluid.layers.elementwise_max(prw, trw)
        cth = fluid.layers.elementwise_max(pth, tth)
        cbh = fluid.layers.elementwise_max(pbh, tbh)
        area_inter = (ilw + irw) * (ith + ibh)
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
            loss = 0.0 - fluid.layers.log(ious)
        else:
            raise KeyError
        if weights is not None:
            loss = loss * weights
        return loss

    def __call__(self, cls_logits, bboxes_reg, centerness, tag_labels,
                 tag_bboxes, tag_center):
        """
        Calculate the loss for classification, location and centerness
        Args:
            cls_logits (list): list of Variables, which is predicted
                score for all anchor points with shape [N, M, C]
            bboxes_reg (list): list of Variables, which is predicted
                offsets for all anchor points with shape [N, M, 4]
            centerness (list): list of Variables, which is predicted
                centerness for all anchor points with shape [N, M, 1]
            tag_labels (list): list of Variables, which is category
                targets for each anchor point
            tag_bboxes (list): list of Variables, which is bounding
                boxes targets for positive samples
            tag_center (list): list of Variables, which is centerness
                targets for positive samples
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
                self.__flatten_tensor(cls_logits[num_lvl - 1 - lvl], True))
            bboxes_reg_flatten_list.append(
                self.__flatten_tensor(bboxes_reg[num_lvl - 1 - lvl], True))
            centerness_flatten_list.append(
                self.__flatten_tensor(centerness[num_lvl - 1 - lvl], True))
            tag_labels_flatten_list.append(
                self.__flatten_tensor(tag_labels[lvl], False))
            tag_bboxes_flatten_list.append(
                self.__flatten_tensor(tag_bboxes[lvl], False))
            tag_center_flatten_list.append(
                self.__flatten_tensor(tag_center[lvl], False))

        cls_logits_flatten = fluid.layers.concat(
            cls_logits_flatten_list, axis=0)
        bboxes_reg_flatten = fluid.layers.concat(
            bboxes_reg_flatten_list, axis=0)
        centerness_flatten = fluid.layers.concat(
            centerness_flatten_list, axis=0)
        tag_labels_flatten = fluid.layers.concat(
            tag_labels_flatten_list, axis=0)
        tag_bboxes_flatten = fluid.layers.concat(
            tag_bboxes_flatten_list, axis=0)
        tag_center_flatten = fluid.layers.concat(
            tag_center_flatten_list, axis=0)
        tag_labels_flatten.stop_gradient = True
        tag_bboxes_flatten.stop_gradient = True
        tag_center_flatten.stop_gradient = True

        mask_positive = tag_labels_flatten > 0
        mask_positive.stop_gradient = True
        mask_positive_float = fluid.layers.cast(mask_positive, dtype="float32")
        mask_positive_float.stop_gradient = True
        num_positive_fp32 = fluid.layers.reduce_sum(mask_positive_float)
        num_positive_int32 = fluid.layers.cast(num_positive_fp32, dtype="int32")
        num_positive_int32 = num_positive_int32 * 0 + 1
        num_positive_fp32.stop_gradient = True
        num_positive_int32.stop_gradient = True
        normalize_sum = fluid.layers.sum(tag_center_flatten)
        normalize_sum.stop_gradient = True
        normalize_sum = fluid.layers.reduce_sum(mask_positive_float * normalize_sum)
        normalize_sum.stop_gradient = True
        cls_loss = fluid.layers.sigmoid_focal_loss(
            cls_logits_flatten, tag_labels_flatten,
            num_positive_int32) / num_positive_fp32
        reg_loss = self.__iou_loss(
            bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float,
            tag_center_flatten) * mask_positive_float / normalize_sum
        ctn_loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=centerness_flatten,
            label=tag_center_flatten) * mask_positive_float / num_positive_fp32
        # loss_all = {
        #     "loss_centerness": fluid.layers.reduce_sum(ctn_loss),
        #     "loss_cls": fluid.layers.reduce_sum(cls_loss),
        #     "loss_box": fluid.layers.reduce_sum(reg_loss)
        # }
        # return loss_all
        return ctn_loss, reg_loss



fCOSLossPaddle = FCOSLossPaddle(loss_alpha=0.25, loss_gamma=2.0, iou_loss_type="giou", reg_weights=1.0)

# cls_logits里面每个元素是[N, 80, 格子行数, 格子列数]
# bboxes_reg里面每个元素是[N,  4, 格子行数, 格子列数]
# centerness里面每个元素是[N,  1, 格子行数, 格子列数]

# 大感受野
cls_logits0 = P.data(name='cls_logits0', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
bboxes_reg0 = P.data(name='bboxes_reg0', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
centerness0 = P.data(name='centerness0', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')

cls_logits1 = P.data(name='cls_logits1', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
bboxes_reg1 = P.data(name='bboxes_reg1', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
centerness1 = P.data(name='centerness1', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')

cls_logits2 = P.data(name='cls_logits2', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
bboxes_reg2 = P.data(name='bboxes_reg2', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
centerness2 = P.data(name='centerness2', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')

cls_logits3 = P.data(name='cls_logits3', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
bboxes_reg3 = P.data(name='bboxes_reg3', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
centerness3 = P.data(name='centerness3', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')

cls_logits4 = P.data(name='cls_logits4', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
bboxes_reg4 = P.data(name='bboxes_reg4', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
centerness4 = P.data(name='centerness4', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')

cls_logits = [cls_logits0, cls_logits1, cls_logits2, cls_logits3, cls_logits4]
bboxes_reg = [bboxes_reg0, bboxes_reg1, bboxes_reg2, bboxes_reg3, bboxes_reg4]
centerness = [centerness0, centerness1, centerness2, centerness3, centerness4]


tag_labels0 = P.data(name='tag_labels0', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
tag_bboxes0 = P.data(name='tag_bboxes0', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
tag_center0 = P.data(name='tag_center0', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')

tag_labels1 = P.data(name='tag_labels1', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
tag_bboxes1 = P.data(name='tag_bboxes1', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
tag_center1 = P.data(name='tag_center1', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')

tag_labels2 = P.data(name='tag_labels2', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
tag_bboxes2 = P.data(name='tag_bboxes2', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
tag_center2 = P.data(name='tag_center2', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')

tag_labels3 = P.data(name='tag_labels3', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
tag_bboxes3 = P.data(name='tag_bboxes3', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
tag_center3 = P.data(name='tag_center3', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')

tag_labels4 = P.data(name='tag_labels4', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='int32')
tag_bboxes4 = P.data(name='tag_bboxes4', shape=[-1, -1, -1, 4], append_batch_size=False, dtype='float32')
tag_center4 = P.data(name='tag_center4', shape=[-1, -1, -1, 1], append_batch_size=False, dtype='float32')


tag_labels = [tag_labels0, tag_labels1, tag_labels2, tag_labels3, tag_labels4]
tag_bboxes = [tag_bboxes0, tag_bboxes1, tag_bboxes2, tag_bboxes3, tag_bboxes4]
tag_center = [tag_center0, tag_center1, tag_center2, tag_center3, tag_center4]


loss_all = fCOSLossPaddle(cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_center)


# Create an executor using CPU as an example
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())



# cls_logits0_ndarray = np.random.rand(1, 80, 96, 128).astype(np.float32)
# cls_logits1_ndarray = np.random.rand(1, 80, 48, 64).astype(np.float32)
# cls_logits2_ndarray = np.random.rand(1, 80, 24, 32).astype(np.float32)
# cls_logits3_ndarray = np.random.rand(1, 80, 12, 16).astype(np.float32)
# cls_logits4_ndarray = np.random.rand(1, 80, 6, 8).astype(np.float32)
#
# bboxes_reg0_ndarray = np.random.rand(1, 4, 96, 128).astype(np.float32)
# bboxes_reg1_ndarray = np.random.rand(1, 4, 48, 64).astype(np.float32)
# bboxes_reg2_ndarray = np.random.rand(1, 4, 24, 32).astype(np.float32)
# bboxes_reg3_ndarray = np.random.rand(1, 4, 12, 16).astype(np.float32)
# bboxes_reg4_ndarray = np.random.rand(1, 4, 6, 8).astype(np.float32)
#
# centerness0_ndarray = np.random.rand(1, 1, 96, 128).astype(np.float32)
# centerness1_ndarray = np.random.rand(1, 1, 48, 64).astype(np.float32)
# centerness2_ndarray = np.random.rand(1, 1, 24, 32).astype(np.float32)
# centerness3_ndarray = np.random.rand(1, 1, 12, 16).astype(np.float32)
# centerness4_ndarray = np.random.rand(1, 1, 6, 8).astype(np.float32)

cls_logits4_ndarray = np.random.rand(1, 80, 96, 128).astype(np.float32)
cls_logits3_ndarray = np.random.rand(1, 80, 48, 64).astype(np.float32)
cls_logits2_ndarray = np.random.rand(1, 80, 24, 32).astype(np.float32)
cls_logits1_ndarray = np.random.rand(1, 80, 12, 16).astype(np.float32)
cls_logits0_ndarray = np.random.rand(1, 80, 6, 8).astype(np.float32)

bboxes_reg4_ndarray = np.random.rand(1, 4, 96, 128).astype(np.float32)
bboxes_reg3_ndarray = np.random.rand(1, 4, 48, 64).astype(np.float32)
bboxes_reg2_ndarray = np.random.rand(1, 4, 24, 32).astype(np.float32)
bboxes_reg1_ndarray = np.random.rand(1, 4, 12, 16).astype(np.float32)
bboxes_reg0_ndarray = np.random.rand(1, 4, 6, 8).astype(np.float32)

centerness4_ndarray = np.random.rand(1, 1, 96, 128).astype(np.float32)
centerness3_ndarray = np.random.rand(1, 1, 48, 64).astype(np.float32)
centerness2_ndarray = np.random.rand(1, 1, 24, 32).astype(np.float32)
centerness1_ndarray = np.random.rand(1, 1, 12, 16).astype(np.float32)
centerness0_ndarray = np.random.rand(1, 1, 6, 8).astype(np.float32)




dic = np.load('../data.npz')
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
batch_labels3 = dic['batch_labels3']
batch_reg_target3 = dic['batch_reg_target3']
batch_centerness3 = dic['batch_centerness3']
batch_labels4 = dic['batch_labels4']
batch_reg_target4 = dic['batch_reg_target4']
batch_centerness4 = dic['batch_centerness4']


# batch_labels4 = dic['batch_labels0']
# batch_reg_target4 = dic['batch_reg_target0']
# batch_centerness4 = dic['batch_centerness0']
# batch_labels3 = dic['batch_labels1']
# batch_reg_target3 = dic['batch_reg_target1']
# batch_centerness3 = dic['batch_centerness1']
# batch_labels2 = dic['batch_labels2']
# batch_reg_target2 = dic['batch_reg_target2']
# batch_centerness2 = dic['batch_centerness2']
# batch_labels1 = dic['batch_labels3']
# batch_reg_target1 = dic['batch_reg_target3']
# batch_centerness1 = dic['batch_centerness3']
# batch_labels0 = dic['batch_labels4']
# batch_reg_target0 = dic['batch_reg_target4']
# batch_centerness0 = dic['batch_centerness4']


print()



results = exe.run(fluid.default_main_program(),
                  feed={'cls_logits0': cls_logits0_ndarray,
                        'cls_logits1': cls_logits1_ndarray,
                        'cls_logits2': cls_logits2_ndarray,
                        'cls_logits3': cls_logits3_ndarray,
                        'cls_logits4': cls_logits4_ndarray,
                        'bboxes_reg0': bboxes_reg0_ndarray,
                        'bboxes_reg1': bboxes_reg1_ndarray,
                        'bboxes_reg2': bboxes_reg2_ndarray,
                        'bboxes_reg3': bboxes_reg3_ndarray,
                        'bboxes_reg4': bboxes_reg4_ndarray,
                        'centerness0': centerness0_ndarray,
                        'centerness1': centerness1_ndarray,
                        'centerness2': centerness2_ndarray,
                        'centerness3': centerness3_ndarray,
                        'centerness4': centerness4_ndarray,
                        'tag_labels0': batch_labels0,
                        'tag_labels1': batch_labels1,
                        'tag_labels2': batch_labels2,
                        'tag_labels3': batch_labels3,
                        'tag_labels4': batch_labels4,
                        'tag_bboxes0': batch_reg_target0,
                        'tag_bboxes1': batch_reg_target1,
                        'tag_bboxes2': batch_reg_target2,
                        'tag_bboxes3': batch_reg_target3,
                        'tag_bboxes4': batch_reg_target4,
                        'tag_center0': batch_centerness0,
                        'tag_center1': batch_centerness1,
                        'tag_center2': batch_centerness2,
                        'tag_center3': batch_centerness3,
                        'tag_center4': batch_centerness4, },
                  fetch_list=[loss_all[0], loss_all[1]])
aaa00 = results[0]
aaa01 = results[1]

print()





# Pytorch实现



class FCOSLossPytorch(torch.nn.Module):
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
        super(FCOSLossPytorch, self).__init__()
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
        focal_loss = focal_loss / (fg_num + 1e-9)
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

        mask_positive = tag_labels_flatten > 0   # [批大小*所有格子数,  1]， 正样本处为True
        mask_positive_float = mask_positive.float()
        num_positive_fp32 = mask_positive_float.sum()   # 这一批的正样本数
        normalize_sum = tag_center_flatten + 0
        normalize_sum = (mask_positive_float * normalize_sum).sum()

        cls_loss = self.sigmoid_focal_loss(cls_logits_flatten, tag_labels_flatten, num_positive_fp32)
        reg_loss = self.__iou_loss(bboxes_reg_flatten, tag_bboxes_flatten, mask_positive_float, tag_center_flatten) \
                   * mask_positive_float / normalize_sum
        ctn_loss = self.sigmoid_cross_entropy_with_logits(
            x=centerness_flatten,
            label=tag_center_flatten) * mask_positive_float / num_positive_fp32
        # loss_all = {
        #     "loss_centerness": fluid.layers.reduce_sum(ctn_loss),
        #     "loss_cls": fluid.layers.reduce_sum(cls_loss),
        #     "loss_box": fluid.layers.reduce_sum(reg_loss)
        # }
        # return loss_all
        return ctn_loss, reg_loss


fCOSLossPytorch = FCOSLossPytorch(loss_alpha=0.25, loss_gamma=2.0, iou_loss_type="giou", reg_weights=1.0)




cls_logits4_ndarray2 = torch.Tensor(cls_logits4_ndarray)
bboxes_reg4_ndarray2 = torch.Tensor(bboxes_reg4_ndarray)
centerness4_ndarray2 = torch.Tensor(centerness4_ndarray)

cls_logits3_ndarray2 = torch.Tensor(cls_logits3_ndarray)
bboxes_reg3_ndarray2 = torch.Tensor(bboxes_reg3_ndarray)
centerness3_ndarray2 = torch.Tensor(centerness3_ndarray)

cls_logits2_ndarray2 = torch.Tensor(cls_logits2_ndarray)
bboxes_reg2_ndarray2 = torch.Tensor(bboxes_reg2_ndarray)
centerness2_ndarray2 = torch.Tensor(centerness2_ndarray)

cls_logits1_ndarray2 = torch.Tensor(cls_logits1_ndarray)
bboxes_reg1_ndarray2 = torch.Tensor(bboxes_reg1_ndarray)
centerness1_ndarray2 = torch.Tensor(centerness1_ndarray)

cls_logits0_ndarray2 = torch.Tensor(cls_logits0_ndarray)
bboxes_reg0_ndarray2 = torch.Tensor(bboxes_reg0_ndarray)
centerness0_ndarray2 = torch.Tensor(centerness0_ndarray)

batch_labels0p = torch.Tensor(batch_labels0)
batch_reg_target0p = torch.Tensor(batch_reg_target0)
batch_centerness0p = torch.Tensor(batch_centerness0)

batch_labels1p = torch.Tensor(batch_labels1)
batch_reg_target1p = torch.Tensor(batch_reg_target1)
batch_centerness1p = torch.Tensor(batch_centerness1)

batch_labels2p = torch.Tensor(batch_labels2)
batch_reg_target2p = torch.Tensor(batch_reg_target2)
batch_centerness2p = torch.Tensor(batch_centerness2)

batch_labels3p = torch.Tensor(batch_labels3)
batch_reg_target3p = torch.Tensor(batch_reg_target3)
batch_centerness3p = torch.Tensor(batch_centerness3)

batch_labels4p = torch.Tensor(batch_labels4)
batch_reg_target4p = torch.Tensor(batch_reg_target4)
batch_centerness4p = torch.Tensor(batch_centerness4)





cls_logits = [cls_logits0_ndarray2, cls_logits1_ndarray2, cls_logits2_ndarray2, cls_logits3_ndarray2, cls_logits4_ndarray2]
bboxes_reg = [bboxes_reg0_ndarray2, bboxes_reg1_ndarray2, bboxes_reg2_ndarray2, bboxes_reg3_ndarray2, bboxes_reg4_ndarray2]
centerness = [centerness0_ndarray2, centerness1_ndarray2, centerness2_ndarray2, centerness3_ndarray2, centerness4_ndarray2]

tag_labels = [batch_labels0p, batch_labels1p, batch_labels2p, batch_labels3p, batch_labels4p]
tag_bboxes = [batch_reg_target0p, batch_reg_target1p, batch_reg_target2p, batch_reg_target3p, batch_reg_target4p]
tag_center = [batch_centerness0p, batch_centerness1p, batch_centerness2p, batch_centerness3p, batch_centerness4p]


bbb00, bbb01 = fCOSLossPytorch(cls_logits, bboxes_reg, centerness, tag_labels, tag_bboxes, tag_center)
aaa02 = bbb00.cpu().detach().numpy()
aaa03 = bbb01.cpu().detach().numpy()

d0 = np.sum((aaa00 - aaa02) ** 2)
d1 = np.sum((aaa01 - aaa03) ** 2)

print()



