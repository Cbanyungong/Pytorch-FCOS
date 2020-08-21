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
import paddle.fluid.layers as P

from model.custom_layers import Conv2dUnit

class ConvBlock(object):
    def __init__(self, filters, use_dcn=False, stride=2, block_name=''):
        '''
        官方SOLO仓库中，下采样是在中间的3x3卷积层进行。
        '''
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, stride=1, padding=0, bias_attr=False, bn=1, act='relu', name=block_name+'.conv0')
        self.conv2 = Conv2dUnit(filters2, 3, stride=stride, padding=1, bias_attr=False, bn=1, act='relu', name=block_name+'.conv1', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, stride=1, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'.conv2')

        self.conv4 = Conv2dUnit(filters3, 1, stride=stride, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'.conv3')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = P.elementwise_add(x=x, y=shortcut, act=None)
        x = P.relu(x)
        return x


class IdentityBlock(object):
    def __init__(self, filters, use_dcn=False, block_name=''):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(filters1, 1, stride=1, padding=0, bias_attr=False, bn=1, act='relu', name=block_name+'.conv0')
        self.conv2 = Conv2dUnit(filters2, 3, stride=1, padding=1, bias_attr=False, bn=1, act='relu', name=block_name+'.conv1', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters3, 1, stride=1, padding=0, bias_attr=False, bn=1, act=None, name=block_name+'.conv2')

    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = P.elementwise_add(x=x, y=input_tensor, act=None)
        x = P.relu(x)
        return x

class Resnet(object):
    def __init__(self, depth, use_dcn=False):
        super(Resnet, self).__init__()
        assert depth in [50, 101]
        self.depth = depth
        self.use_dcn = use_dcn
        self.conv1 = Conv2dUnit(64, 7, stride=2, padding=3, bias_attr=False, bn=1, act='relu', name='backbone.stage1.0.conv0')

        # stage2
        self.stage2_0 = ConvBlock([64, 64, 256], stride=1, block_name='backbone.stage2.0')
        self.stage2_1 = IdentityBlock([64, 64, 256], block_name='backbone.stage2.1')
        self.stage2_2 = IdentityBlock([64, 64, 256], block_name='backbone.stage2.2')

        # stage3
        self.stage3_0 = ConvBlock([128, 128, 512], block_name='backbone.stage3.0', use_dcn=use_dcn)
        self.stage3_1 = IdentityBlock([128, 128, 512], block_name='backbone.stage3.1', use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock([128, 128, 512], block_name='backbone.stage3.2', use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock([128, 128, 512], block_name='backbone.stage3.3', use_dcn=use_dcn)

        # stage4
        self.stage4_0 = ConvBlock([256, 256, 1024], block_name='backbone.stage4.0', use_dcn=use_dcn)
        k = 21
        if depth == 50:
            k = 4
        self.stage4_layers = []
        p = 1
        for i in range(k):
            ly = IdentityBlock([256, 256, 1024], block_name='backbone.stage4.%d' % p, use_dcn=use_dcn)
            self.stage4_layers.append(ly)
            p += 1
        self.stage4_last_layer = IdentityBlock([256, 256, 1024], block_name='backbone.stage4.%d' % p, use_dcn=use_dcn)

        # stage5
        self.stage5_0 = ConvBlock([512, 512, 2048], block_name='backbone.stage5.0', use_dcn=use_dcn)
        self.stage5_1 = IdentityBlock([512, 512, 2048], block_name='backbone.stage5.1', use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock([512, 512, 2048], block_name='backbone.stage5.2', use_dcn=use_dcn)


    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = fluid.layers.pool2d(
            input=x,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        # stage2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        s4 = self.stage2_2(x)
        # stage3
        x = self.stage3_0(s4)
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        s8 = self.stage3_3(x)
        # stage4
        x = self.stage4_0(s8)
        for ly in self.stage4_layers:
            x = ly(x)
        s16 = self.stage4_last_layer(x)
        # stage5
        x = self.stage5_0(s16)
        x = self.stage5_1(x)
        s32 = self.stage5_2(x)
        return [s4, s8, s16, s32]


