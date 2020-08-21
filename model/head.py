#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-23 15:12:37
#   Description : paddlepaddle_solo
#
# ================================================================
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as P
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from model.custom_layers import Conv2dUnit


def concat_coord(x):
    ins_feat = x  # [N, c, h, w]

    batch_size = P.shape(x)[0]
    h = P.shape(x)[2]
    w = P.shape(x)[3]
    float_h = P.cast(h, 'float32')
    float_w = P.cast(w, 'float32')

    y_range = P.range(0., float_h, 1., dtype='float32')     # [h, ]
    y_range = 2.0 * y_range / (float_h - 1.0) - 1.0
    x_range = P.range(0., float_w, 1., dtype='float32')     # [w, ]
    x_range = 2.0 * x_range / (float_w - 1.0) - 1.0
    x_range = P.reshape(x_range, (1, -1))   # [1, w]
    y_range = P.reshape(y_range, (-1, 1))   # [h, 1]
    x = P.expand(x_range, [h, 1])     # [h, w]
    y = P.expand(y_range, [1, w])     # [h, w]

    x = P.reshape(x, (1, 1, h, w))   # [1, 1, h, w]
    y = P.reshape(y, (1, 1, h, w))   # [1, 1, h, w]
    x = P.expand(x, [batch_size, 1, 1, 1])   # [N, 1, h, w]
    y = P.expand(y, [batch_size, 1, 1, 1])   # [N, 1, h, w]

    ins_feat_x = P.concat([ins_feat, x], axis=1)   # [N, c+1, h, w]
    ins_feat_y = P.concat([ins_feat, y], axis=1)   # [N, c+1, h, w]

    return [ins_feat_x, ins_feat_y]

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = P.pool2d(heat, pool_size=kernel, pool_stride=1,
                    pool_padding=[[0, 0], [0, 0], [1, 0], [1, 0]],
                    pool_type='max')
    keep = P.cast(P.equal(hmax, heat), 'float32')
    return heat * keep


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)   0、1组成的掩码
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor):  shape (n, )      n个物体的面积

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = P.shape(cate_labels)[0]   # 物体数
    seg_masks = P.reshape(seg_masks, (n_samples, -1))   # [n, h*w]
    # inter.
    inter_matrix = P.matmul(seg_masks, seg_masks, transpose_y=True)   # [n, n] 自己乘以自己的转置。两两之间的交集面积。
    # union.
    sum_masks_x = P.expand(P.reshape(sum_masks, (1, -1)), [n_samples, 1])     # [n, n]  sum_masks重复了n行得到sum_masks_x
    # iou.
    iou_matrix = inter_matrix / (sum_masks_x + P.transpose(sum_masks_x, [1, 0]) - inter_matrix)
    rows = P.range(0, n_samples, 1, 'int32')
    cols = P.range(0, n_samples, 1, 'int32')
    rows = P.expand(P.reshape(rows, (1, -1)), [n_samples, 1])
    cols = P.expand(P.reshape(cols, (-1, 1)), [1, n_samples])
    tri_mask = P.cast(rows > cols, 'float32')
    iou_matrix = tri_mask * iou_matrix   # [n, n]   只取上三角部分

    # label_specific matrix.
    cate_labels_x = P.expand(P.reshape(cate_labels, (1, -1)), [n_samples, 1])     # [n, n]  cate_labels重复了n行得到cate_labels_x
    label_matrix = P.cast(P.equal(cate_labels_x, P.transpose(cate_labels_x, [1, 0])), 'float32')
    label_matrix = tri_mask * label_matrix   # [n, n]   只取上三角部分

    # IoU compensation
    compensate_iou = P.reduce_max(iou_matrix * label_matrix, dim=0)
    compensate_iou = P.expand(P.reshape(compensate_iou, (1, -1)), [n_samples, 1])     # [n, n]
    compensate_iou = P.transpose(compensate_iou, [1, 0])      # [n, n]

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # # matrix nms
    if kernel == 'gaussian':
        decay_matrix = P.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = P.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient = P.reduce_min((decay_matrix / compensate_matrix), dim=0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient = P.reduce_min(decay_matrix, dim=0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


class DecoupledSOLOHead(object):
    def __init__(self,
                 num_classes=80,
                 in_channels=256,
                 seg_feat_channels=256,
                 stacked_convs=7,
                 strides=[8, 8, 16, 32, 32],
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
                 sigma=0.2,
                 num_grids=[40, 36, 24, 16, 12],
                 cate_down_pos=0,
                 with_deform=False,
                 loss_ins=None,
                 loss_cate=None):
        super(DecoupledSOLOHead, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.sigma = sigma
        self.cate_down_pos = cate_down_pos
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.with_deform = with_deform
        # self.loss_cate = build_loss(loss_cate)
        # self.ins_loss_weight = loss_ins['loss_weight']
        self._init_layers()

    def _init_layers(self):
        self.ins_convs_x = []
        self.ins_convs_y = []
        self.cate_convs = []

        for i in range(self.stacked_convs):
            conv2d_1 = Conv2dUnit(self.seg_feat_channels, 3, stride=1, padding=1, bias_attr=False, gn=1, groups=32, act='relu', name='bbox_head.ins_convs_x.%d' % i)
            self.ins_convs_x.append(conv2d_1)

            conv2d_2 = Conv2dUnit(self.seg_feat_channels, 3, stride=1, padding=1, bias_attr=False, gn=1, groups=32, act='relu', name='bbox_head.ins_convs_y.%d' % i)
            self.ins_convs_y.append(conv2d_2)

            conv2d_3 = Conv2dUnit(self.seg_feat_channels, 3, stride=1, padding=1, bias_attr=False, gn=1, groups=32, act='relu', name='bbox_head.cate_convs.%d' % i)
            self.cate_convs.append(conv2d_3)

        self.dsolo_ins_list_x = []
        self.dsolo_ins_list_y = []
        for i, seg_num_grid in enumerate(self.seg_num_grids):
            conv2d_1 = Conv2dUnit(seg_num_grid, 3, stride=1, padding=1, bias_attr=True, name='bbox_head.dsolo_ins_list_x.%d' % i)
            self.dsolo_ins_list_x.append(conv2d_1)
            conv2d_2 = Conv2dUnit(seg_num_grid, 3, stride=1, padding=1, bias_attr=True, name='bbox_head.dsolo_ins_list_y.%d' % i)
            self.dsolo_ins_list_y.append(conv2d_2)
        self.dsolo_cate = Conv2dUnit(self.num_classes, 3, stride=1, padding=1, bias_attr=True, name='head.dsolo_cate')

    def __call__(self, feats, cfg, eval):
        # DecoupledSOLOHead都是这样，一定有5个张量，5个张量的strides=[8, 8, 16, 32, 32]，所以先对首尾张量进行插值。
        new_feats = [P.resize_bilinear(feats[0], out_shape=P.shape(feats[1])[2:]),
                     feats[1],
                     feats[2],
                     feats[3],
                     P.resize_bilinear(feats[4], out_shape=P.shape(feats[3])[2:])]
        featmap_sizes = [P.shape(featmap)[2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0] * 2, featmap_sizes[0][1] * 2)   # stride=4

        ins_pred_x_list, ins_pred_y_list, cate_pred_list = [], [], []
        for idx in range(len(self.seg_num_grids)):
            ins_feat = new_feats[idx]   # 给掩码分支
            cate_feat = new_feats[idx]  # 给分类分支

            # ============ ins branch (掩码分支，特征图形状是[N, grid, mask_h, mask_w]) ============
            ins_feat_x, ins_feat_y = concat_coord(ins_feat)   # [N, c+1, h, w]、 [N, c+1, h, w]

            for ins_layer_x, ins_layer_y in zip(self.ins_convs_x, self.ins_convs_y):
                ins_feat_x = ins_layer_x(ins_feat_x)   # [N, 256, h, w]
                ins_feat_y = ins_layer_y(ins_feat_y)   # [N, 256, h, w]

            ins_feat_x = P.resize_bilinear(ins_feat_x, scale=float(2))   # [N, 256, 2*h, 2*w]
            ins_feat_y = P.resize_bilinear(ins_feat_y, scale=float(2))   # [N, 256, 2*h, 2*w]

            ins_pred_x = self.dsolo_ins_list_x[idx](ins_feat_x)   # [N, grid, 2*h, 2*w]，即[N, grid, mask_h, mask_w]
            ins_pred_y = self.dsolo_ins_list_y[idx](ins_feat_y)   # [N, grid, 2*h, 2*w]，即[N, grid, mask_h, mask_w]
            # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
            # 那么对应的ins_pred_x大小应该为[104, 104, 52, 26, 26]；
            # 那么对应的ins_pred_y大小应该为[104, 104, 52, 26, 26]。

            # ============ cate branch (分类分支，特征图形状是[N, num_classes=80, grid, grid]) ============
            for i, cate_layer in enumerate(self.cate_convs):
                if i == self.cate_down_pos:   # 第0次都要插值成seg_num_grid x seg_num_grid的大小。
                    seg_num_grid = self.seg_num_grids[idx]
                    cate_feat = P.resize_bilinear(cate_feat, out_shape=(seg_num_grid, seg_num_grid))
                cate_feat = cate_layer(cate_feat)

            cate_pred = self.dsolo_cate(cate_feat)   # 种类分支，通道数变成了80，[N, 80, grid, grid]

            # ============ 是否是预测状态 ============
            if eval:
                ins_pred_x = P.sigmoid(ins_pred_x)
                ins_pred_x = P.resize_bilinear(ins_pred_x, out_shape=upsampled_size)

                ins_pred_y = P.sigmoid(ins_pred_y)
                ins_pred_y = P.resize_bilinear(ins_pred_y, out_shape=upsampled_size)
                # 若输入图片大小为416x416，那么new_feats里图片大小应该为[52, 52, 26, 13, 13]，因为strides=[8, 8, 16, 32, 32]。
                # 那么此处的5个ins_pred_x大小应该为[104, 104, 104, 104, 104]；
                # 那么此处的5个ins_pred_y大小应该为[104, 104, 104, 104, 104]。即stride=4。训练时不会执行这里。
                cate_pred = P.sigmoid(cate_pred)
                cate_pred = points_nms(cate_pred)
            ins_pred_x_list.append(ins_pred_x)
            ins_pred_y_list.append(ins_pred_y)
            cate_pred_list.append(cate_pred)
        if eval:
            num_layers = len(self.seg_num_grids)
            pred_cate = []
            for i in range(num_layers):
                c = cate_pred_list[i]   # 从小感受野 到 大感受野 （从多格子 到 少格子）
                c = P.transpose(c, perm=[0, 2, 3, 1])
                c = P.reshape(c, (1, -1, self.num_classes))
                pred_cate.append(c)
            pred_mask_x = P.concat(ins_pred_x_list, axis=1)
            pred_mask_y = P.concat(ins_pred_y_list, axis=1)
            pred_cate = P.concat(pred_cate, axis=1)
            output = self.get_seg_single(pred_cate[0],
                   pred_mask_x[0],
                   pred_mask_y[0],
                   upsampled_size,
                   upsampled_size,
                   cfg)
            return output
        return ins_pred_x_list + ins_pred_y_list + cate_pred_list

    def get_seg_single(self,
                       cate_preds,
                       seg_preds_x,
                       seg_preds_y,
                       featmap_size,
                       ori_shape,
                       cfg):
        '''
        Args:
            cate_preds:    同一张图片5个输出层的输出汇合  [40*40+36*36+24*24+16*16+12*12, 80]
            seg_preds_x:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            seg_preds_y:   同一张图片5个输出层的输出汇合  [40+36+24+16+12, 104, 104]
            featmap_size:  [s4, s4]        一维张量  1-D Tensor
            img_shape:     [800, 1216, 3]  一维张量  1-D Tensor
            ori_shape:     [427, 640, 3]   一维张量  1-D Tensor
            scale_factor:  800/427
            cfg:
            rescale:
            debug:

        Returns:

        '''
        # trans trans_diff.
        seg_num_grids = P.assign(np.array(self.seg_num_grids))
        trans_size = P.cumsum(P.pow(seg_num_grids, 2))
        seg_size = P.cumsum(seg_num_grids)    # [40, 40+36, 40+36+24, ...]

        trans_diff = []
        seg_diff = []
        num_grids = []
        strides = []
        n_stage = len(self.seg_num_grids)   # 5个输出层
        for ind_ in range(n_stage):
            if ind_ == 0:
                # 第0个输出层的分类分支在cate_preds中的偏移是0
                trans_diff_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'int32')
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是0
                seg_diff_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'int32')
            else:
                # 第1个输出层的分类分支在cate_preds中的偏移是40*40，第2个输出层的分类分支在cate_preds中的偏移是40*40+36*36，...
                trans_diff_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'int32') + trans_size[ind_ - 1]
                # 第0个输出层的掩码分支在seg_preds_x中的偏移是40，第0个输出层的掩码分支在seg_preds_x中的偏移是40+36，...
                seg_diff_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'int32') + seg_size[ind_ - 1]
            # 第0个输出层的一行（或一列）的num_grids是40，第1个输出层的一行（或一列）的num_grids是36，...
            num_grids_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'int32') + self.seg_num_grids[ind_]
            # 第0个输出层的stride是8，第1个输出层的stride是8，...
            strides_ = P.zeros([self.seg_num_grids[ind_] ** 2, ], 'float32') + float(self.strides[ind_])

            trans_diff.append(trans_diff_)
            seg_diff.append(seg_diff_)
            num_grids.append(num_grids_)
            strides.append(strides_)
        trans_diff = P.concat(trans_diff, axis=0)   # [3872, ]
        seg_diff = P.concat(seg_diff, axis=0)       # [3872, ]
        num_grids = P.concat(num_grids, axis=0)     # [3872, ]
        strides = P.concat(strides, axis=0)         # [3872, ]

        # process. 处理。
        inds = P.where(cate_preds > cfg.score_thr)   # [[3623, 17], [3623, 60], [3639, 17], ...]   分数超过阈值的物体所在格子
        inds_extra = P.zeros([1, 2], 'int64')
        inds = P.concat([inds, inds_extra], axis=0)
        cate_scores = P.gather_nd(cate_preds, inds)

        trans_diff = P.gather(trans_diff, inds[:, 0])   # [3472, 3472, 3472, ...]   格子所在输出层的分类分支在cate_preds中的偏移
        seg_diff = P.gather(seg_diff, inds[:, 0])       # [100, 100, 100, ...]      格子所在输出层的掩码分支在seg_preds_x中的偏移
        num_grids = P.gather(num_grids, inds[:, 0])     # [16, 16, 16, ...]         格子所在输出层每一行有多少个格子
        strides = P.gather(strides, inds[:, 0])         # [32, 32, 32, ...]         格子所在输出层的stride

        loc = P.cast(inds[:, 0], 'int32')
        y_inds = (loc - trans_diff) // num_grids   # 格子行号
        x_inds = (loc - trans_diff) % num_grids    # 格子列号
        y_inds += seg_diff   # 格子行号在seg_preds_y中的绝对位置
        x_inds += seg_diff   # 格子列号在seg_preds_x中的绝对位置

        cate_labels = inds[:, 1]   # 类别
        mask_x = P.gather(seg_preds_x, x_inds)   # [11, s4, s4]
        mask_y = P.gather(seg_preds_y, y_inds)   # [11, s4, s4]
        seg_masks_soft = mask_x * mask_y    # [11, s4, s4]  物体的mask，逐元素相乘得到。
        seg_masks = P.cast(seg_masks_soft > cfg.mask_thr, 'float32')
        sum_masks = P.reduce_sum(seg_masks, dim=[1, 2])   # [11, ]  11个物体的面积
        keep = P.where(sum_masks > strides)   # 面积大于这一层的stride才保留

        def exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels, keep):
            seg_masks_soft = P.gather_nd(seg_masks_soft, keep)  # 用概率表示的掩码
            seg_masks = P.gather_nd(seg_masks, keep)  # 用True、False表示的掩码
            cate_scores = P.gather_nd(cate_scores, keep)  # 类别得分
            sum_masks = P.gather_nd(sum_masks, keep)  # 面积
            cate_labels = P.gather_nd(cate_labels, keep)  # 类别
            # mask scoring   是1的像素的 概率总和 占 面积（是1的像素数） 的比重
            seg_score = P.reduce_sum(seg_masks_soft * seg_masks, dim=[1, 2]) / sum_masks
            cate_scores *= seg_score  # 类别得分乘上这个比重得到新的类别得分。因为有了mask scoring机制，所以分数一般比其它算法如yolact少。


            # sort and keep top nms_pre
            _, sort_inds = P.argsort(cate_scores, axis=-1, descending=True)  # [7, 5, 8, ...] 降序。最大值的下标，第2大值的下标，...
            sort_inds = sort_inds[:cfg.nms_pre]  # 最多cfg.nms_pre个。
            seg_masks_soft = P.gather(seg_masks_soft, sort_inds)  # 按照分数降序
            seg_masks = P.gather(seg_masks, sort_inds)            # 按照分数降序
            cate_scores = P.gather(cate_scores, sort_inds)
            sum_masks = P.gather(sum_masks, sort_inds)
            cate_labels = P.gather(cate_labels, sort_inds)

            # Matrix NMS
            cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores,
                                     kernel=cfg.kernel, sigma=cfg.sigma, sum_masks=sum_masks)

            keep = P.where(cate_scores > cfg.update_thr)   # 大于第二个分数阈值才保留

            def exist_objs_2(cate_scores, seg_masks_soft, cate_labels, keep):
                keep = P.reshape(keep, (-1,))
                seg_masks_soft = P.gather(seg_masks_soft, keep)
                cate_scores = P.gather(cate_scores, keep)
                cate_labels = P.gather(cate_labels, keep)

                # sort and keep top_k
                _, sort_inds = P.argsort(cate_scores, axis=-1, descending=True)
                sort_inds = sort_inds[:cfg.max_per_img]
                seg_masks_soft = P.gather(seg_masks_soft, sort_inds)
                cate_scores = P.gather(cate_scores, sort_inds)
                cate_labels = P.gather(cate_labels, sort_inds)

                # 插值前处理
                seg_masks_soft = P.unsqueeze(seg_masks_soft, axes=[0])

                # seg_masks_soft = tf.image.resize_images(seg_masks_soft, tf.convert_to_tensor([featmap_size[0] * 4, featmap_size[1] * 4]), method=tf.image.ResizeMethod.BILINEAR)
                # seg_masks = tf.image.resize_images(seg_masks_soft, tf.convert_to_tensor([ori_shape[0], ori_shape[1]]), method=tf.image.ResizeMethod.BILINEAR)

                seg_masks_soft = P.resize_bilinear(seg_masks_soft, out_shape=[featmap_size[0] * 4, featmap_size[1] * 4])
                seg_masks = P.resize_bilinear(seg_masks_soft, out_shape=[ori_shape[0], ori_shape[1]])

                # 插值后处理
                seg_masks = P.cast(seg_masks > cfg.mask_thr, 'float32')
                cate_labels = P.reshape(cate_labels, (1, -1))
                cate_scores = P.reshape(cate_scores, (1, -1))
                return seg_masks, cate_labels, cate_scores


            def no_objs_2():
                seg_masks = P.zeros([1, 1, 1, 1], 'float32') - 1.0
                cate_labels = P.zeros([1, 1], 'int64') - 1
                cate_scores = P.zeros([1, 1], 'float32') - 1.0
                return seg_masks, cate_labels, cate_scores

            # 是否有物体
            seg_masks, cate_labels, cate_scores = P.cond(P.shape(keep)[0] == 0,
                                                         no_objs_2,
                                                         lambda: exist_objs_2(cate_scores, seg_masks_soft, cate_labels, keep))
            return seg_masks, cate_labels, cate_scores

        def no_objs_1():
            seg_masks = P.zeros([1, 1, 1, 1], 'float32') - 1.0
            cate_labels = P.zeros([1, 1], 'int64') - 1
            cate_scores = P.zeros([1, 1], 'float32') - 1.0
            return seg_masks, cate_labels, cate_scores

        # 是否有物体
        seg_masks, cate_labels, cate_scores = P.cond(P.shape(keep)[0] == 0,
                                                     no_objs_1,
                                                     lambda: exist_objs_1(cate_scores, seg_masks_soft, seg_masks, sum_masks, cate_labels, keep))
        return [seg_masks, cate_labels, cate_scores]


