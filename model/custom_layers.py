#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-23 15:12:37
#   Description : paddlepaddle_solo
#
# ================================================================
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


class Conv2dUnit(object):
    def __init__(self,
                 filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 groups=32,
                 act=None,
                 name='',
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 use_dcn=False):
        super(Conv2dUnit, self).__init__()
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.bias_attr = bias_attr
        self.bn = bn
        self.gn = gn
        self.groups = groups
        self.act = act
        self.name = name
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn

    def __call__(self, x):
        conv_name = self.name + ".conv"
        if self.use_dcn:
            pass
        else:
            battr = None
            if self.bias_attr:
                battr = ParamAttr(name=conv_name + ".bias")
            x = fluid.layers.conv2d(
                input=x,
                num_filters=self.filters,
                filter_size=self.filter_size,
                stride=self.stride,
                padding=self.padding,
                act=None,
                param_attr=ParamAttr(name=conv_name + ".weights"),
                bias_attr=battr,
                name=conv_name + '.output.1')
        if self.bn:
            bn_name = self.name + ".bn"
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=bn_name + '.scale',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            battr = ParamAttr(
                name=bn_name + '.offset',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            x = fluid.layers.batch_norm(
                input=x,
                name=bn_name + '.output.1',
                is_test=self.is_test,  # 冻结层时（即trainable=False），bn的均值、标准差也还是会变化，只有设置is_test=True才保证不变
                param_attr=pattr,
                bias_attr=battr,
                moving_mean_name=bn_name + '.mean',
                moving_variance_name=bn_name + '.var')
        if self.gn:
            gn_name = self.name + ".gn"
            norm_lr = 0. if self.freeze_norm else 1.   # 归一化层学习率
            norm_decay = self.norm_decay   # 衰减
            pattr = ParamAttr(
                name=gn_name + '.scale',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            battr = ParamAttr(
                name=gn_name + '.offset',
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))   # L2权重衰减正则化
            x = fluid.layers.group_norm(
                input=x,
                groups=self.groups,
                name=gn_name + '.output.1',
                param_attr=pattr,
                bias_attr=battr)
        if self.act == 'leaky':
            x = fluid.layers.leaky_relu(x, alpha=0.1)
        elif self.act == 'relu':
            x = fluid.layers.relu(x)
        return x




