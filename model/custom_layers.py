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
import torch.nn.functional as F


class AffineChannel(torch.nn.Module):
    def __init__(self, num_features):
        super(AffineChannel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_features, ))
        self.bias = torch.nn.Parameter(torch.randn(num_features, ))

    def forward(self, x):
        N = x.size(0)
        C = x.size(1)
        H = x.size(2)
        W = x.size(3)
        transpose_x = x.permute(0, 2, 3, 1)
        flatten_x = transpose_x.reshape(N*H*W, C)
        out = flatten_x * self.weight + self.bias
        out = out.reshape(N, H, W, C)
        out = out.permute(0, 3, 1, 2)
        return out


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Conv2dUnit(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 filters,
                 filter_size,
                 stride=1,
                 bias_attr=False,
                 bn=0,
                 gn=0,
                 af=0,
                 groups=32,
                 act=None,
                 freeze_norm=False,
                 is_test=False,
                 norm_decay=0.,
                 use_dcn=False):
        super(Conv2dUnit, self).__init__()
        self.groups = groups
        self.act = act
        self.freeze_norm = freeze_norm
        self.is_test = is_test
        self.norm_decay = norm_decay
        self.use_dcn = use_dcn

        # conv
        if use_dcn:
            pass
        else:
            self.conv = torch.nn.Conv2d(input_dim, filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, bias=bias_attr)

        # norm
        self.bn = None
        self.gn = None
        self.af = None
        if bn:
            self.bn = torch.nn.BatchNorm2d(filters)
        if gn:
            self.gn = torch.nn.GroupNorm(num_groups=groups, num_channels=filters)
        if af:
            self.af = AffineChannel(filters)

        # act
        self.act = None
        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky':
            self.act = torch.nn.LeakyReLU(0.1)
        elif act == 'mish':
            self.act = Mish()

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.gn:
            x = self.gn(x)
        if self.act:
            x = self.act(x)
        return x





