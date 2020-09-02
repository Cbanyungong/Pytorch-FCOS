#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos。读取AdelaiDet FCOS_RT_MS_DLA_34_4x的权重。
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

state_dict = load_weights('FCOS_RT_MS_DLA_34_4x_syncbn.pth')
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



cfg = FCOS_RT_DLA34_FPN_4x_Config()

# 创建模型
Backbone = select_backbone(cfg.backbone_type)
backbone = Backbone(**cfg.backbone)

Fpn = select_fpn(cfg.fpn_type)
fpn = Fpn(**cfg.fpn)

Loss = select_loss(cfg.fcos_loss_type)
fcos_loss = Loss(**cfg.fcos_loss)

Head = select_head(cfg.head_type)
head = Head(num_classes=80, fcos_loss=fcos_loss, **cfg.head)

fcos = FCOS(backbone, fpn, head)

print('\nCopying...')


def copy_conv_bn(conv_unit, w, scale, offset, m, v):
    conv, bn = conv_unit.conv, conv_unit.bn
    conv.weight.data = torch.Tensor(w)
    bn.weight.data = torch.Tensor(scale)
    bn.bias.data = torch.Tensor(offset)
    bn.running_mean.data = torch.Tensor(m)
    bn.running_var.data = torch.Tensor(v)

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

# dla34
w = backbone_dic['backbone.bottom_up.base_layer.0.weight']
scale = backbone_dic['backbone.bottom_up.base_layer.1.weight']
offset = backbone_dic['backbone.bottom_up.base_layer.1.bias']
m = backbone_dic['backbone.bottom_up.base_layer.1.running_mean']
v = backbone_dic['backbone.bottom_up.base_layer.1.running_var']
copy_conv_bn(backbone.base_layer, w, scale, offset, m, v)

w = backbone_dic['backbone.bottom_up.level0.0.weight']
scale = backbone_dic['backbone.bottom_up.level0.1.weight']
offset = backbone_dic['backbone.bottom_up.level0.1.bias']
m = backbone_dic['backbone.bottom_up.level0.1.running_mean']
v = backbone_dic['backbone.bottom_up.level0.1.running_var']
copy_conv_bn(backbone.level0[0], w, scale, offset, m, v)


w = backbone_dic['backbone.bottom_up.level1.0.weight']
scale = backbone_dic['backbone.bottom_up.level1.1.weight']
offset = backbone_dic['backbone.bottom_up.level1.1.bias']
m = backbone_dic['backbone.bottom_up.level1.1.running_mean']
v = backbone_dic['backbone.bottom_up.level1.1.running_var']
copy_conv_bn(backbone.level1[0], w, scale, offset, m, v)



print()


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
num_convs = 4
ids = [[0, 1], [3, 4], [6, 7], [9, 10]]
for lvl in range(0, num_convs):
    # conv + gn
    w = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][0]]
    b = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][0]]
    scale = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.weight'%ids[lvl][1]]
    offset = fcos_head_dic['proposal_generator.fcos_head.cls_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn(head.cls_convs[lvl], w, b, scale, offset)


    # conv + gn
    w = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][0]]
    b = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][0]]
    scale = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.weight'%ids[lvl][1]]
    offset = fcos_head_dic['proposal_generator.fcos_head.bbox_tower.%d.bias'%ids[lvl][1]]
    copy_conv_gn(head.reg_convs[lvl], w, b, scale, offset)

# 类别分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.cls_logits.weight']
b = fcos_head_dic['proposal_generator.fcos_head.cls_logits.bias']
copy_conv(head.cls_convs[-1], w, b)

# 坐标分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.weight']
b = fcos_head_dic['proposal_generator.fcos_head.bbox_pred.bias']
copy_conv(head.reg_convs[-1], w, b)

# centerness分支最后的conv
w = fcos_head_dic['proposal_generator.fcos_head.ctrness.weight']
b = fcos_head_dic['proposal_generator.fcos_head.ctrness.bias']
copy_conv(head.ctn_conv, w, b)

# 3个scale。请注意，AdelaiDet在head部分是从小感受野到大感受野遍历，而PaddleDetection是从大感受野到小感受野遍历。所以这里scale顺序反过来。
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.0.scale']
head.scales_on_reg[2].data = torch.Tensor(scale_i)
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.1.scale']
head.scales_on_reg[1].data = torch.Tensor(scale_i)
scale_i = fcos_head_dic['proposal_generator.fcos_head.scales.2.scale']
head.scales_on_reg[0].data = torch.Tensor(scale_i)



torch.save(fcos.state_dict(), 'fcos_rt_dla34_fpn_4x.pt')
print('\nDone.')


