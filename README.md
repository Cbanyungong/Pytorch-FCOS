[English](README_en.md) | 简体中文

# Pytorch-FCOS

## 概述
Pytorch-FCOS,参考自https://github.com/PaddlePaddle/PaddleDetection

## 咩酱刷屏时刻

Keras版YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch版YOLOv3：https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle版YOLOv3：https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle完美复刻版版yolact: https://github.com/miemie2013/PaddlePaddle_yolact

yolov3魔改成yolact: https://github.com/miemie2013/yolact

Keras版YOLOv4: https://github.com/miemie2013/Keras-YOLOv4 (mAP 41%+)

Pytorch版YOLOv4: https://github.com/miemie2013/Pytorch-YOLOv4 (mAP 41%+)

Paddle版YOLOv4：https://github.com/miemie2013/Paddle-YOLOv4 (mAP 41%+)

Keras版SOLO: https://github.com/miemie2013/Keras-SOLO

Paddle版SOLO: https://github.com/miemie2013/Paddle-SOLO

Pytorch版FCOS: https://github.com/miemie2013/Pytorch-FCOS

## 更新日记

2020/08/21:开工。

## 需要补充

更多模型。

## 环境搭建

requirements.txt

## 训练
下载我从PaddleDetection保存下来的模型fcos_r50_fpn_multiscale_2x.npz
链接：https://pan.baidu.com/s/1qEcsQyKm2fk-ECBrWrH8HA
提取码：cq31

将它放在项目根目录下。然后运行1_paddle_fcos_r50_fpn_multiscale_2x2pytorch.py得到一个fcos_r50_fpn_multiscale_2x.pt，它也位于根目录下。
运行train.py进行训练。通过修改config.py代码来进行更换数据集、更改超参数以及训练参数。

训练时如果发现mAP很稳定了，就停掉，修改学习率为原来的十分之一，接着继续训练，mAP还会再上升。暂时是这样手动操作。

## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path这5个变量（自带的voc2012数据集直接解除注释就ok了）就可以开始训练自己的数据集了。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。该mAP是val集的结果。
fcos_r50_fpn_multiscale_2x.pt在COCO2017 val下的mAP如下：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.604
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.247
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.327
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.538
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.389
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
```

fcos_rt_r50_fpn_4x.pt在COCO2017 val下的mAP如下：
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.577
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.533
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.317
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.608
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.710
```


## test-dev
运行test_dev.py。
运行完之后，进入results目录，把bbox_detections.json压缩成bbox_detections.zip，提交到
https://competitions.codalab.org/competitions/20794#participate
获得bbox mAP.

上述pytorch_yolov4.pt在test集的mAP是（input_shape = (608, 608)，分数阈值=0.001，nms阈值=0.45的情况下）
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.xxx
```

该mAP是test集的结果，也就是大部分检测算法论文的标准指标。

## 预测
运行demo.py。

## 预测视频
运行demo_video.py。（按esc键停止播放）

## 传送门
cv算法交流q群：645796480
但是关于仓库的疑问尽量在Issues上提，避免重复解答。

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

## 广告位招租
有偿接私活，可联系微信wer186259，金主快点来吧！
