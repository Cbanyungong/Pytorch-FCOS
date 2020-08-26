#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================
import torch

from model.custom_layers import Conv2dUnit

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, filters, bn, gn, af, use_dcn=False, stride=2):
        '''
        ResNetVB的下采样是在中间的3x3卷积层进行。
        '''
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=1, bn=bn, gn=gn, af=af, act='relu')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=stride, bn=bn, gn=gn, af=af, act='relu', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, act=None)

        self.conv4 = Conv2dUnit(in_c, filters3, 1, stride=stride, bn=bn, gn=gn, af=af, act=None)

        self.act = torch.nn.ReLU()

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        shortcut = self.conv4(input_tensor)
        x = x + shortcut
        x = self.act(x)
        return x


class IdentityBlock(torch.nn.Module):
    def __init__(self, in_c, filters, bn, gn, af, use_dcn=False):
        super(IdentityBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv1 = Conv2dUnit(in_c,     filters1, 1, stride=1, bn=bn, gn=gn, af=af, act='relu')
        self.conv2 = Conv2dUnit(filters1, filters2, 3, stride=1, bn=bn, gn=gn, af=af, act='relu', use_dcn=use_dcn)
        self.conv3 = Conv2dUnit(filters2, filters3, 1, stride=1, bn=bn, gn=gn, af=af, act=None)

        self.act = torch.nn.ReLU()

    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + input_tensor
        x = self.act(x)
        return x

class Resnet(torch.nn.Module):
    def __init__(self, depth, norm_type='affine_channel', feature_maps=[3, 4, 5], use_dcn=False):
        super(Resnet, self).__init__()
        assert depth in [50, 101]
        self.depth = depth
        self.norm_type = norm_type
        self.feature_maps = feature_maps
        self.use_dcn = use_dcn

        bn = 0
        gn = 0
        af = 0
        if norm_type == 'bn':
            bn = 1
        elif norm_type == 'gn':
            gn = 1
        elif norm_type == 'affine_channel':
            af = 1
        self.conv1 = Conv2dUnit(3, 64, 7, stride=2, bn=bn, gn=gn, af=af, act='relu')
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # stage2
        self.stage2_0 = ConvBlock(64, [64, 64, 256], bn, gn, af, stride=1)
        self.stage2_1 = IdentityBlock(256, [64, 64, 256], bn, gn, af)
        self.stage2_2 = IdentityBlock(256, [64, 64, 256], bn, gn, af)

        # stage3
        self.stage3_0 = ConvBlock(256, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)
        self.stage3_1 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)
        self.stage3_2 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)
        self.stage3_3 = IdentityBlock(512, [128, 128, 512], bn, gn, af, use_dcn=use_dcn)

        # stage4
        self.stage4_0 = ConvBlock(512, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
        k = 21
        if depth == 50:
            k = 4
        self.stage4_layers = []
        p = 1
        for i in range(k):
            ly = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)
            self.stage4_layers.append(ly)
            p += 1
        self.stage4_last_layer = IdentityBlock(1024, [256, 256, 1024], bn, gn, af, use_dcn=use_dcn)

        # stage5
        self.stage5_0 = ConvBlock(1024, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn)
        self.stage5_1 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn)
        self.stage5_2 = IdentityBlock(2048, [512, 512, 2048], bn, gn, af, use_dcn=use_dcn)

    def to_cuda(self, device=None):
        for i, ly in enumerate(self.stage4_layers):
            self.stage4_layers[i] = self.stage4_layers[i].cuda(device)
        self2 = self.cuda(device)
        return self2


    def forward(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool(x)

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

        outs = []
        if 2 in self.feature_maps:
            outs.append(s4)
        if 3 in self.feature_maps:
            outs.append(s8)
        if 4 in self.feature_maps:
            outs.append(s16)
        if 5 in self.feature_maps:
            outs.append(s32)
        return outs


