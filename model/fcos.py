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

    def __call__(self, x, cfg, eval):
        x = self.backbone(x)
        x = self.neck(x)
        if eval:
            x = self.head(x, cfg.test_cfg, eval)
        else:
            x = self.head(x, None, eval)
        return x




