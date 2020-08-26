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



def _compute_locations(feature):
    shape_fm = fluid.layers.shape(feature)
    shape_fm.stop_gradient = True
    h = shape_fm[2]
    w = shape_fm[3]
    # fpn_stride = self.fpn_stride[lvl]
    fpn_stride = 128
    shift_x = fluid.layers.range(
        0, w * fpn_stride, fpn_stride, dtype='float32')   # 生成x偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
    shift_y = fluid.layers.range(
        0, h * fpn_stride, fpn_stride, dtype='float32')   # 生成y偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
    shift_x = fluid.layers.unsqueeze(shift_x, axes=[0])   # [1, w]
    shift_y = fluid.layers.unsqueeze(shift_y, axes=[1])   # [h, 1]
    shift_x = fluid.layers.expand_as(
        shift_x, target_tensor=feature[0, 0, :, :])       # [h, w]
    shift_y = fluid.layers.expand_as(
        shift_y, target_tensor=feature[0, 0, :, :])       # [h, w]
    shift_x.stop_gradient = True
    shift_y.stop_gradient = True
    shift_x = fluid.layers.reshape(shift_x, shape=[-1])   # [h*w, ]
    shift_y = fluid.layers.reshape(shift_y, shape=[-1])   # [h*w, ]
    location = fluid.layers.stack(
        [shift_x, shift_y], axis=-1) + fpn_stride // 2    # [h*w, 2]  格子中心点的坐标
    location.stop_gradient = True

    return location


feature = P.data(name='feature', shape=[-1, 256, -1, -1], append_batch_size=False, dtype='float32')
location = _compute_locations(feature)


# Create an executor using CPU as an example
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

feature_ndarray = np.random.rand(2, 256, 5, 7).astype(np.float32)

results = exe.run(fluid.default_main_program(),
                  feed={'feature': feature_ndarray, }, fetch_list=[location])
aaa00 = results[0]

print()





# Pytorch实现
def _compute_locations2(feature):
    shape_fm = feature.size()
    h = shape_fm[2]
    w = shape_fm[3]
    fpn_stride = 128
    shift_x = torch.arange(0, w, dtype=torch.float32) * fpn_stride   # 生成x偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
    shift_y = torch.arange(0, h, dtype=torch.float32) * fpn_stride   # 生成y偏移 [0, 1*fpn_stride, 2*fpn_stride, ...]
    shift_x = shift_x.unsqueeze(0)   # [1, w]
    shift_y = shift_y.unsqueeze(1)   # [h, 1]
    shift_x = shift_x.repeat((h, 1))   # [h, w]
    shift_y = shift_y.repeat((1, w))   # [h, w]
    shift_x = shift_x.reshape((h*w, 1))   # [h*w, 1]
    shift_y = shift_y.reshape((h*w, 1))   # [h*w, 1]
    location = torch.cat([shift_x, shift_y], dim=-1)   # [h*w, 2]  格子左上角的坐标，单位是1像素。顺序是先第一行格子从左到右，再到第二行格子从左到右，...
    location += fpn_stride // 2                        # [h*w, 2]  格子中心点的坐标，单位是1像素。顺序是先第一行格子从左到右，再到第二行格子从左到右，...
    return location


feature2 = torch.Tensor(feature_ndarray)

aaa01 = _compute_locations2(feature2)
aaa01 = aaa01.cpu().detach().numpy()

print()



