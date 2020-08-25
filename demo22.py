#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-08-19 17:20:11
#   Description : pytorch_yolov4
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import numpy as np
import torch

from model.custom_layers import Conv2dUnit
from model.fcos import FCOS
from model.head import FCOSHead
from model.neck import FPN
from model.resnet import Resnet

import platform
sysstr = platform.system()
use_cuda = torch.cuda.is_available()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False



net = Resnet(50)
fpn = FPN(256)
head = FCOSHead()
# fcos = FCOS()


# print(net)
# print(fpn)
print(head)



conv_layer = Conv2dUnit(256, 256, 3, stride=1, bias_attr=True, gn=1, groups=32, act='relu')



print(conv_layer)











