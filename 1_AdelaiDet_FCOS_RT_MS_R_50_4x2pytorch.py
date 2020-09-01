#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos。读取AdelaiDet FCOS_RT_MS_R_50_4x的权重。
#
# ================================================================
import numpy as np
from paddle import fluid
import torch

from config import *
from model.fcos import *
from model.head import *
from model.neck import *
from model.resnet import *




def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('FCOS_RT_MS_R_50_4x_syncbn.pth')
print('============================================================')

backbone_dic = {}
fpn_dic = {}
fcos_head_dic = {}
others = {}
for key, value in state_dict.items():
    if 'tracked' in key:
        continue
    if 'bottom_up' in key:
        backbone_dic[key] = value.data.numpy()
    elif 'fpn' in key:
        fpn_dic[key] = value.data.numpy()
    elif 'fcos_head' in key:
        fcos_head_dic[key] = value.data.numpy()
    else:
        others[key] = value.data.numpy()

print()



cfg = FCOS_RT_R50_FPN_4x_Config()

resnet = Resnet(**cfg.resnet)
fpn = FPN(**cfg.fpn)
head = FCOSSharedHead(num_classes=80, **cfg.head)
fcos = FCOS(resnet, fpn, head)


print('\nCopying...')


def copy_conv_af(conv_unit, w, scale, offset):
    conv, af = conv_unit.conv, conv_unit.af
    conv.weight.data = torch.Tensor(w)
    af.weight.data = torch.Tensor(scale)
    af.bias.data = torch.Tensor(offset)

def copy_conv(conv_unit, w, b):
    conv = conv_unit.conv
    conv.weight.data = torch.Tensor(w)
    conv.bias.data = torch.Tensor(b)

def copy_conv_gn(conv_unit, w, b, scale, offset):
    conv, gn = conv_unit.conv, conv_unit.gn
    conv.weight.data = torch.Tensor(w)
    conv.bias.data = torch.Tensor(b)
    gn.weight.data = torch.Tensor(scale)
    gn.bias.data = torch.Tensor(offset)



# 获取FCOS模型的权重


# Resnet50
w = dic['conv1_weights']
scale = dic['bn_conv1_scale']
offset = dic['bn_conv1_offset']
copy_conv_af(resnet.conv1, w, scale, offset)


nums = [3, 4, 6, 3]
for nid, num in enumerate(nums):
    stage_name = 'res' + str(nid + 2)
    for kk in range(num):
        block_name = stage_name + chr(ord("a") + kk)
        conv_name1 = block_name + "_branch2a"
        conv_name2 = block_name + "_branch2b"
        conv_name3 = block_name + "_branch2c"
        shortcut_name = block_name + "_branch1"

        bn_name1 = 'bn' + conv_name1[3:]
        bn_name2 = 'bn' + conv_name2[3:]
        bn_name3 = 'bn' + conv_name3[3:]
        shortcut_bn_name = 'bn' + shortcut_name[3:]

        w = dic[conv_name1 + '_weights']
        scale = dic[bn_name1 + '_scale']
        offset = dic[bn_name1 + '_offset']
        copy_conv_af(resnet.get_block('stage%d_%d' % (2+nid, kk)).conv1, w, scale, offset)

        w = dic[conv_name2 + '_weights']
        scale = dic[bn_name2 + '_scale']
        offset = dic[bn_name2 + '_offset']
        copy_conv_af(resnet.get_block('stage%d_%d' % (2+nid, kk)).conv2, w, scale, offset)

        w = dic[conv_name3 + '_weights']
        scale = dic[bn_name3 + '_scale']
        offset = dic[bn_name3 + '_offset']
        copy_conv_af(resnet.get_block('stage%d_%d' % (2+nid, kk)).conv3, w, scale, offset)

        # 每个stage的第一个卷积块才有4个卷积层
        if kk == 0:
            w = dic[shortcut_name + '_weights']
            scale = dic[shortcut_bn_name + '_scale']
            offset = dic[shortcut_bn_name + '_offset']
            copy_conv_af(resnet.get_block('stage%d_%d' % (2+nid, kk)).conv4, w, scale, offset)
# fpn, 6个卷积层
w = fpn_dic['backbone.fpn_lateral5.weight']
b = fpn_dic['backbone.fpn_lateral5.bias']
copy_conv(fpn.s32_conv, w, b)

w = fpn_dic['backbone.fpn_lateral4.weight']
b = fpn_dic['backbone.fpn_lateral4.bias']
copy_conv(fpn.s16_conv, w, b)

w = fpn_dic['backbone.fpn_lateral3.weight']
b = fpn_dic['backbone.fpn_lateral3.bias']
copy_conv(fpn.s8_conv, w, b)

w = fpn_dic['backbone.fpn_output5.weight']
b = fpn_dic['backbone.fpn_output5.bias']
copy_conv(fpn.sc_s32_conv, w, b)

w = fpn_dic['backbone.fpn_output4.weight']
b = fpn_dic['backbone.fpn_output4.bias']
copy_conv(fpn.sc_s16_conv, w, b)

w = fpn_dic['backbone.fpn_output3.weight']
b = fpn_dic['backbone.fpn_output3.bias']
copy_conv(fpn.sc_s8_conv, w, b)


# head
n = 3  # 有n个输出层
num_convs = 4
ids = [[0, 1], [3, 4], [6, 7], [9, 10]]
for i in range(n):  # 遍历每个输出层
    for lvl in range(0, num_convs):
        # conv + gn
        w = fcos_head_dic['proposal_generator.fcos_head.cls_tower.0.weight']
        b = fcos_head_dic['proposal_generator.fcos_head.cls_tower.0.bias']
        scale = fcos_head_dic['proposal_generator.fcos_head.cls_tower.1.weight']
        offset = fcos_head_dic['proposal_generator.fcos_head.cls_tower.1.bias']
        copy_conv_gn(head.cls_convs_per_feature[i][lvl], w, b, scale, offset)


        # conv + gn
        conv_reg_name = 'fcos_head_reg_tower_conv_{}'.format(lvl)
        norm_name = conv_reg_name + "_norm"
        w = dic[conv_reg_name + "_weights"]
        b = dic[conv_reg_name + "_bias"]
        scale = dic[norm_name + "_scale"]
        offset = dic[norm_name + "_offset"]
        copy_conv_gn(head.reg_convs_per_feature[i][lvl], w, b, scale, offset)

    # 类别分支最后的conv
    conv_cls_name = "fcos_head_cls"
    w = dic[conv_cls_name + "_weights"]
    b = dic[conv_cls_name + "_bias"]
    copy_conv(head.cls_convs_per_feature[i][-1], w, b)

    # 坐标分支最后的conv
    conv_reg_name = "fcos_head_reg"
    w = dic[conv_reg_name + "_weights"]
    b = dic[conv_reg_name + "_bias"]
    copy_conv(head.reg_convs_per_feature[i][-1], w, b)

    # centerness分支最后的conv
    conv_centerness_name = "fcos_head_centerness"
    w = dic[conv_centerness_name + "_weights"]
    b = dic[conv_centerness_name + "_bias"]
    copy_conv(head.ctn_convs_per_feature[i], w, b)

# 5个scale
fpn_names = ['fpn_7', 'fpn_6', 'fpn_res5_sum', 'fpn_res4_sum', 'fpn_res3_sum']
i = 0
for fpn_name in fpn_names:
    scale_i = dic["%s_scale_on_reg" % fpn_name]
    head.scales_on_reg[i].data = torch.Tensor(scale_i)
    i += 1



torch.save(fcos.state_dict(), 'fcos_r50_fpn_multiscale_2x.pt')
print('\nDone.')


