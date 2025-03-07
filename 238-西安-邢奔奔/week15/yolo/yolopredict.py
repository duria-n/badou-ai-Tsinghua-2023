import colorsys
import os.path
import random

import numpy as np
import tensorflow as tf

import config
from yolov3 import yolo


class yolo_predictor:
    def __init__(self,obj_threshold,nms_threshold,classes_file,anchors_file):
        """
            初始化函数
        :param obj_threshold:
        :param nms_threshold:
        :param classes_file:
        :param anchors_file:
        """
        self.obj_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.classes_path = classes_file
        self.anchors_path = anchors_file
        #获取先验框和类别名
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        #画框
        hsv_tuples = [(x / len(self.class_names),1.,1.) for x in range(len(self.class_names))]
        #给每个类别设置属于自己的颜色
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors = list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),
                                         self.colors))
        random.seed(1001)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        #获取类别名称
        class_path = os.path.expanduser(self.classes_path)
        with open(class_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        #获取anchors坐标
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f :
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape((-1,2))

        return anchors

    def boxes_and_scores(self,feats,anchors,classes_num,input_shape,image_shape):
        """
                解码数据
        :param feats: yolo输出的特征层数据
        :param anchors: anchor位置
        :param classes_num: 类别数量
        :param input_shape: 输入尺寸
        :param image_shape: 图像形状
        :return:
            boxes：物体框的位置
            boxes_scores：物体框置信度和类别的乘积
        """
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats,anchors,classes_num,input_shape)
        boxes = self.correct_boxes(box_xy,box_wh,input_shape,image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        #框的置信度和每个类别的概率相乘
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores,[-1,classes_num])
        return boxes, box_scores

    def correct_boxes(self,box_xy,box_wh,input_shape,image_shape):
        """
            解码物体框的实际位置
        :param box_xy: 左上角坐标
        :param box_wh: 宽高
        :param input_shape:
        :param image_shape:
        :return:
            boxes：物体框框的具体信息
        """
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]

        input_shape = tf.cast(input_shape,dtype=tf.float32)
        image_shape = tf.cast(image_shape,dtype=tf.float32)
        #tf.round对结果进行四舍五入的取整，使用reduce.min来获得计算比例结果中的较小值，来波啊张不会超出界限
        new_shape = tf.round(image_shape*tf.reduce_min(input_shape/image_shape))
        offset = (input_shape-new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw/2.)
        box_maxes = box_yx +(box_hw/2.)
        boxes = tf.concat([box_mins[...,0:1],box_mins[...,1:2],box_maxes[...,0:1],box_maxes[...,1:2]],axis=-1)
        #(y_min, x_min, y_max, x_max)，将框的归一化坐标信息乘图像的形状，获取对应的坐标信息
        boxes *= tf.concat([image_shape,image_shape],axis=-1)
        return boxes


    def _get_feats(self,feats,anchors,num_classes,input_shape):
        """
            解码过程，将神经网络的输出转为锚框的位置和类别信息
        :param feats: yolo模型的最后一层输出
        :param anchors: anchors的位置
        :param num_classes:
        :param input_shape:
        :return:
            物体框的信息
        """
        num_anchors = len(anchors)
        #这里anchor_tensors表示的是锚框信息，最后的2维度为锚框坐标或者宽高
        anchors_tensors = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,num_anchors,2])
        grid_size = tf.shape(feats)[1:3]
        #predicitons是预测结果，shape为[batch_size，h，w，锚框数量，锚框信息],其中num_class+5为x,y,h,w,confidence,class_probs)
        predictions = tf.reshape(feats,[-1,grid_size[0],grid_size[1],num_anchors,num_classes+5])
        #tf.tile对张量进行复制和扩展，这里的目的是生成网格格点坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]),[-1,1,1,1]),[1,grid_size[1],1,1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]),[1,-1,1,1]),[grid_size[0],1,1,1])
        grid = tf.concat([grid_x,grid_y],axis=-1)
        grid = tf.cast(grid,tf.float32)
        #这里就是对bbox的结果做sigmod确保输出值在0，1之间
        """
            将相对偏移映射为针对全图的坐标
        """
        box_xy = (tf.sigmoid(predictions[...,:2])+grid) / tf.cast(grid_size[::-1],tf.float32)
        box_wh = tf.exp(predictions[...,2:4])* anchors_tensors / tf.cast(input_shape[::-1],tf.float32)
        box_confidence = tf.sigmoid(predictions[...,4:5])
        box_class_probs = tf.sigmoid(predictions[...,5:])
        return box_xy, box_wh, box_confidence,box_class_probs
    def eval(self,yolo_outputs,image_shape,max_boxes=20):
        """
            yolo模型的输出做非极大值抑制一直，获取物体检测框的相关信息
        :param yolo_outputs:
        :param image_shape:
        :param max_boxes:
        :return:
        """
        #每个特征层对应的先验框索引
        anchor_mask = [[6,7,8],[3,4,5],[0,1,2]]
        boxes = []
        box_scores = []
        #这里是为了将inputshape转换回输入的尺寸，即416
        input_shape = tf.shape(yolo_outputs[0])[1:3]*32
        #对三个特征层输出进行解码 ，获取预测框的坐标和位置
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i],self.anchors[anchor_mask[i]],
                                                          len(self.class_names),input_shape,image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes,axis=0)
        box_scores = tf.concat(box_scores,axis=0)
        #取出每一类大于阈值的框和分数
        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes,dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        #对于每一个类进行判断
        for c in range(len(self.class_names)):
            """
                取出所有类为c的box
                取出所有类为c的分数
                进行非极大值抑制
                获取非极大值抑制抑制结果
            """
            class_boxes = tf.boolean_mask(boxes,mask[:,c])

            class_box_scores = tf.boolean_mask(box_scores[:,c],mask[:,c])

            nms_index = tf.image.non_max_suppression(class_boxes,class_box_scores,max_boxes_tensor,iou_threshold=self.nms_threshold)

            class_boxes = tf.gather(class_boxes,nms_index)
            class_box_scores = tf.gather(class_box_scores,nms_index)
            classes = tf.ones_like(class_box_scores,'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_,axis=0)
        scores_ = tf.concat(scores_,axis=0)
        classes_ = tf.concat(classes_,axis=0)
        return boxes_, scores_, classes_

    def predict(self,inputs,image_shape):
        """
            predict预测分三步：
            建立yolo对象，获得预测结果
            对预测结果进行处理
        :param inputs:
        :param image_shape:
        :return:
        """

        model = yolo(config.norm_epsilon,config.norm_decay,self.anchors_path,self.classes_path,pre_train=False)
        #output接受三个层次特征网络输出
        output = model.yolo_inference(inputs,config.num_anchors//3,config.num_classes,training=False)
        boxes, scores, classes = self.eval(output,image_shape,max_boxes=20)
        return boxes, scores, classes
