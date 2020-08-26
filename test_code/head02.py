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


def __merge_hw(input, ch_type="channel_first"):
    """
    Args:
        input (Variables): Feature map whose H and W will be merged into one dimension
        ch_type     (str): channel_first / channel_last
    Return:
        new_shape (Variables): 返回H维和W维合并成一维之后张量的新形状
    """
    shape_ = fluid.layers.shape(input)
    bs = shape_[0]
    ch = shape_[1]
    hi = shape_[2]
    wi = shape_[3]
    img_size = hi * wi
    img_size.stop_gradient = True
    if ch_type == "channel_first":
        new_shape = fluid.layers.concat([bs, ch, img_size])
    elif ch_type == "channel_last":
        new_shape = fluid.layers.concat([bs, img_size, ch])
    else:
        raise KeyError("Wrong ch_type %s" % ch_type)
    new_shape.stop_gradient = True
    return new_shape


def _postprocessing_by_level(locations, box_cls, box_reg, box_ctn,
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
    batch_size = 2
    num_classes = 80
    act_shape_cls = __merge_hw(box_cls)
    box_cls_ch_last = fluid.layers.reshape(   # [N, 80, H*W]
        x=box_cls,
        shape=[batch_size, num_classes, -1],
        actual_shape=act_shape_cls)
    box_cls_ch_last = fluid.layers.sigmoid(box_cls_ch_last)   # 类别概率用sigmoid()激活，[N, 80, H*W]
    act_shape_reg = __merge_hw(box_reg, "channel_last")
    box_reg_ch_last = fluid.layers.transpose(box_reg, perm=[0, 2, 3, 1])   # [N, H, W, 4]
    box_reg_ch_last = fluid.layers.reshape(   # [N, H*W, 4]，坐标不用再接激活层，直接预测。
        x=box_reg_ch_last,
        shape=[batch_size, -1, 4],
        actual_shape=act_shape_reg)
    act_shape_ctn = __merge_hw(box_ctn)
    box_ctn_ch_last = fluid.layers.reshape(   # [N, 1, H*W]
        x=box_ctn,
        shape=[batch_size, 1, -1],
        actual_shape=act_shape_ctn)
    box_ctn_ch_last = fluid.layers.sigmoid(box_ctn_ch_last)   # centerness用sigmoid()激活，[N, 1, H*W]

    box_reg_decoding = fluid.layers.stack(
        [
            locations[:, 0] - box_reg_ch_last[:, :, 0],   # 左上角x坐标
            locations[:, 1] - box_reg_ch_last[:, :, 1],   # 左上角y坐标
            locations[:, 0] + box_reg_ch_last[:, :, 2],   # 右下角x坐标
            locations[:, 1] + box_reg_ch_last[:, :, 3]    # 右下角y坐标
        ],
        axis=1)
    box_reg_decoding = fluid.layers.transpose(
        box_reg_decoding, perm=[0, 2, 1])
    # # recover the location to original image
    im_scale = im_info[:, 2]
    box_reg_decoding = box_reg_decoding / im_scale
    box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last
    return box_cls_ch_last, box_reg_decoding



location = P.data(name='location', shape=[-1, 2], append_batch_size=False, dtype='float32')
box_cls = P.data(name='box_cls', shape=[-1, 80, -1, -1], append_batch_size=False, dtype='float32')
box_reg = P.data(name='box_reg', shape=[-1, 4, -1, -1], append_batch_size=False, dtype='float32')
box_ctn = P.data(name='box_ctn', shape=[-1, 1, -1, -1], append_batch_size=False, dtype='float32')
im_info = P.data(name='im_info', shape=[-1, 3], append_batch_size=False, dtype='float32')

box_cls_ch_last, box_reg_decoding = _postprocessing_by_level(location, box_cls, box_reg, box_ctn, im_info)

# Create an executor using CPU as an example
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

location_ndarray = np.array([[64, 64], [192, 64], [64, 192], [192, 192]]).astype(np.float32)
box_cls_ndarray = np.random.rand(2, 80, 2, 2).astype(np.float32)
box_reg_ndarray = np.random.rand(2, 4, 2, 2).astype(np.float32)
box_ctn_ndarray = np.random.rand(2, 1, 2, 2).astype(np.float32)
im_info_ndarray = np.array([[800, 800, 1], [1024, 1024, 1]]).astype(np.float32)

# box_cls_ndarray = np.random.rand(1, 80, 2, 2).astype(np.float32)
# box_reg_ndarray = np.random.rand(1, 4, 2, 2).astype(np.float32)
# box_ctn_ndarray = np.random.rand(1, 1, 2, 2).astype(np.float32)
# im_info_ndarray = np.array([[800, 800, 1]]).astype(np.float32)

results = exe.run(fluid.default_main_program(),
                  feed={'location': location_ndarray, 'box_cls': box_cls_ndarray, 'box_reg': box_reg_ndarray, 'box_ctn': box_ctn_ndarray, 'im_info': im_info_ndarray, },
                  fetch_list=[box_cls_ch_last, box_reg_decoding])
aaa00 = results[0]
aaa01 = results[1]

print()





# Pytorch实现
def _postprocessing_by_level2(locations, box_cls, box_reg, box_ctn,
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
    batch_size = 2
    num_classes = 80

    # =========== 类别概率，[N, 80, H*W] ===========
    box_cls_ch_last = box_cls.reshape((batch_size, num_classes, box_cls.size()[2] * box_cls.size()[3]))   # [N, 80, H*W]
    box_cls_ch_last = torch.sigmoid(box_cls_ch_last)   # 类别概率用sigmoid()激活，[N, 80, H*W]

    # =========== 坐标(4个偏移)，[N, H*W, 4] ===========
    box_reg_ch_last = box_reg.permute(0, 2, 3, 1)   # [N, H, W, 4]
    box_reg_ch_last = box_reg_ch_last.reshape((batch_size, box_reg_ch_last.size()[1] * box_reg_ch_last.size()[2], 4))   # [N, H*W, 4]，坐标不用再接激活层，直接预测。

    # =========== centerness，[N, 1, H*W] ===========
    box_ctn_ch_last = box_ctn.reshape((batch_size, 1, box_ctn.size()[2] * box_ctn.size()[3]))   # [N, 1, H*W]
    box_ctn_ch_last = torch.sigmoid(box_ctn_ch_last)   # centerness用sigmoid()激活，[N, 1, H*W]

    box_reg_decoding = torch.cat(   # [N, H*W, 4]
        [
            locations[:, 0:1] - box_reg_ch_last[:, :, 0:1],   # 左上角x坐标
            locations[:, 1:2] - box_reg_ch_last[:, :, 1:2],   # 左上角y坐标
            locations[:, 0:1] + box_reg_ch_last[:, :, 2:3],   # 右下角x坐标
            locations[:, 1:2] + box_reg_ch_last[:, :, 3:4]    # 右下角y坐标
        ],
        dim=-1)
    # # recover the location to original image
    im_scale = im_info[:, 2]   # [N, ]
    im_scale = im_scale[:, np.newaxis, np.newaxis]   # [N, 1, 1]
    box_reg_decoding = box_reg_decoding / im_scale  # [N, H*W, 4]，最终坐标=坐标*图片缩放因子
    box_cls_ch_last = box_cls_ch_last * box_ctn_ch_last   # [N, 80, H*W]，最终分数=类别概率*centerness
    return box_cls_ch_last, box_reg_decoding



location_ndarray2 = torch.Tensor(location_ndarray)
box_cls_ndarray2 = torch.Tensor(box_cls_ndarray)
box_reg_ndarray2 = torch.Tensor(box_reg_ndarray)
box_ctn_ndarray2 = torch.Tensor(box_ctn_ndarray)
im_info_ndarray2 = torch.Tensor(im_info_ndarray)

bbb00, bbb01 = _postprocessing_by_level2(location_ndarray2, box_cls_ndarray2, box_reg_ndarray2, box_ctn_ndarray2, im_info_ndarray2)
bbb00 = bbb00.cpu().detach().numpy()
bbb01 = bbb01.cpu().detach().numpy()

d0 = np.sum((aaa00 - bbb00) ** 2)
d1 = np.sum((aaa01 - bbb01) ** 2)

print()



