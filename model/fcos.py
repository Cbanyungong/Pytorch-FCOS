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

class FCOS(torch.nn.Module):
    def __init__(self, backbone, neck, head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def to_cuda(self, device=None):
        self.backbone = self.backbone.to_cuda(device)
        self.head = self.head.to_cuda(device)
        self2 = self.cuda(device)
        return self2

    def forward(self, x, im_info, eval):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.neck(body_feats)
        if eval:
            pred = self.head.get_prediction(body_feats, im_info)
        else:
            pred = None
        return pred




