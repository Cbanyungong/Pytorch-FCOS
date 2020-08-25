#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-21 19:33:37
#   Description : pytorch_fcos
#
# ================================================================

class FCOS(object):
    def __init__(self, backbone, neck, head):
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def __call__(self, x, eval):
        body_feats = self.backbone(x)
        body_feats, spatial_scale = self.neck(body_feats)
        if eval:
            im_info = 1
            pred = self.head.get_prediction(body_feats, im_info)
        else:
            pred = None
        return pred




