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



class FPN(torch.nn.Module):
    def __init__(self,
                 num_chan=256,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(FPN, self).__init__()

        # 对骨干网络的sxx进行卷积
        self.s32_conv = Conv2dUnit(2048, num_chan, 1, stride=1, bias_attr=True, act=None)
        self.s16_conv = Conv2dUnit(1024, num_chan, 1, stride=1, bias_attr=True, act=None)
        self.s8_conv = Conv2dUnit(512, num_chan, 1, stride=1, bias_attr=True, act=None)
        self.convs = [self.s32_conv, self.s16_conv, self.s8_conv]

        # 第二次卷积
        self.sc_s32_conv = Conv2dUnit(num_chan, num_chan, 3, stride=1, bias_attr=True, act=None)
        self.sc_s16_conv = Conv2dUnit(num_chan, num_chan, 3, stride=1, bias_attr=True, act=None)
        self.sc_s8_conv = Conv2dUnit(num_chan, num_chan, 3, stride=1, bias_attr=True, act=None)
        self.second_convs = [self.sc_s32_conv, self.sc_s16_conv, self.sc_s8_conv]

        # c6c7
        self.c6_conv = Conv2dUnit(num_chan, num_chan, 3, stride=2, bias_attr=True, act=None)
        self.c7_conv = Conv2dUnit(num_chan, num_chan, 3, stride=2, bias_attr=True, act=None)

        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, body_feats):
        '''
        :param body_feats:  [s8, s16, s32]
        :return:
        '''
        reverse_body_feats = body_feats[::-1]   #   [s32, s16, s8]

        num_backbone_stages = len(reverse_body_feats)   # 3
        # fpn内部的输出
        fpn_inner_output = [None for _ in range(num_backbone_stages)]

        body_input = reverse_body_feats[0]   # 骨干网络的s32
        fpn_inner_output[0] = self.convs[0](body_input)   # fpn的s32
        for i in range(1, num_backbone_stages):
            body_input = reverse_body_feats[i]     # 骨干网络的s16
            top_output = fpn_inner_output[i - 1]   # fpn的s32

            # 骨干网络的s16卷积，fpn的s32上采样，再融合，融合方式为逐元素相加
            lateral = self.convs[i](body_input)   # 骨干网络的s16卷积，stride=16
            topdown = self.upsample(top_output)   # fpn的s32上采样，stride=16
            fpn_inner_single = lateral + topdown   # fpn的s16
            fpn_inner_output[i] = fpn_inner_single   # fpn的s16


        # 第二次卷积
        fpn_outputs = [None for _ in range(num_backbone_stages)]
        for i in range(num_backbone_stages):
            fpn_input = fpn_inner_output[i]   # fpn的s32
            fpn_output = self.second_convs[i](fpn_input)   # fpn的s32
            fpn_outputs[i] = fpn_output

        # c6c7
        c6_input = fpn_outputs[0]   # fpn的s32
        c6 = self.c6_conv(c6_input)
        c7 = self.c7_conv(c6)

        outs = [c7, c6] + fpn_outputs   # [s128, s64, s32, s16, s8]
        spatial_scale = [1. / 128., 1. / 64., 1. / 32., 1. / 16., 1. / 8.]
        return outs, spatial_scale


