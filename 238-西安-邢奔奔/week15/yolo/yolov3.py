import os.path

import numpy as np
import tensorflow as tf


class yolo:
    def __init__(self,norm_epsilon,norm_decay,anchors_path,class_path,pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.class_path =class_path
        self.anchors_path = anchors_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()


    def _get_class(self):
        class_path = os.path.expanduser(self.class_path)
        with open(class_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1,2)


    #用于生成层

    def _batch_normalization(self,input_layers,name=None,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        bn_layer = tf.layers.batch_normalization(inputs=input_layers,momentum=norm_decay,epsilon=norm_epsilon,
                                                 center=True,scale=True,training=training,name=name)
        return tf.nn.leaky_relu(bn_layer,alpha=0.1)

    def _conv2d_layer(self,inputs,filter_num,kernel_size,name,use_bias=False,strides=1):
        # conv = tf.layers.conv2d(inputs=inputs,filters=filter_num,kernel_size=kernel_size,strides=[strides,strides],
        #                         kernel_initializer=tf.glorot_uniform_initializer(),
        #                         padding=('SAME' if strides == 1 else 'VALID'),kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),use_bias=use_bias,name=name)
        # return conv
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filter_num,
            kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name)
        return conv


    def _Residual_block(self,inputs,filters_num,blocks_num,conv_index,training=True,norm_decay=0.99,
                        norm_epsilon=1e-3):
        """
        :param inputs:
        :param filters_num:
        :param blocks_num:
        :param conv_index:
        :param training:
        :param norm_decay:
        :param norm_epsilon:
        :return:
        """
        inputs = tf.pad(inputs,paddings=[[0,0],[1,0],[1,0],[0,0]],mode='CONSTANT')
        layer = self._conv2d_layer(inputs,filters_num,kernel_size=3,strides=2,name="conv2d_"+str(conv_index))
        layer = self._batch_normalization(layer,name="batch_normalization_"+str(conv_index),training=training,
                                          norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer,filters_num//2,kernel_size=1,strides=1,name="conv2d_"+str(conv_index))
            layer = self._batch_normalization(layer,name="batch_normalization_"+str(conv_index),training=training,norm_epsilon=norm_epsilon,norm_decay=norm_decay)
            conv_index += 1
            layer = self._conv2d_layer(layer,filters_num,kernel_size=3,strides=1,name="conv2d_"+str(conv_index))
            layer = self._batch_normalization(layer,name="batch_normalization_"+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index
    def darknet53(self,input,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        """

        :param input:
        :param conv_index:
        :param training:
        :param norm_decay:
        :param norm_epsilon:
        :return:
        router1:返回26层卷积计算结果，52，52，256
        router2:返回第43层卷积计算结果：26，26，512
        conv_index:卷积层计数
        """
        with tf.variable_scope("darknet53"):

            #416,416,3->416,416,32
            conv = self._conv2d_layer(input,filter_num=32,kernel_size=3,strides=1,name="conv2d_"+str(conv_index))
            conv = self._batch_normalization(conv,name="batch_normalization_"+str(conv_index),training=training,norm_decay = norm_decay,norm_epsilon=norm_epsilon)
            conv_index += 1

            conv, conv_index = self._Residual_block(conv,conv_index=conv_index,filters_num=64,blocks_num=1,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index,filters_num=128,blocks_num=2,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index,filters_num=256,blocks_num=8,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            router1 = conv
            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filters_num=512,blocks_num=8,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            router2 = conv

            conv,conv_index = self._Residual_block(conv,conv_index=conv_index,filters_num=1024,blocks_num=4,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            return router1, router2, conv, conv_index

    def _yolo_block(self,inputs,filters_num,out_filters,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        """

        :param inputs:
        :param filters_num:
        :param out_filters:
        :param conv_index:
        :param training:
        :param norm_decay:
        :param norm_epsilon:
        :return: route返回倒数第二层卷积的结果
        """
        conv = self._conv2d_layer(inputs,filter_num=filters_num,kernel_size=1,strides=1,name="conv2d_"+str(conv_index))
        conv = self._batch_normalization(conv,name="batch_normalization_"+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv,filter_num=filters_num*2,kernel_size=3,strides=1,name="conv2d_"+str(conv_index))
        conv = self._batch_normalization(conv,name="batch_normalization_"+str(conv_index),training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filters_num, kernel_size=1, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization(conv,  name="batch_normalization_" + str(conv_index),training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filters_num * 2, kernel_size=3, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filter_num=filters_num, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filter_num=filters_num * 2, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
        conv = self._batch_normalization(conv, name="batch_normalization_" + str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv,filter_num=out_filters,kernel_size=1,strides=1,name="conv2d_"+str(conv_index),use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self,inputs,num_anchors,num_classes,training=True):
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self.darknet53(inputs,conv_index,training=training,norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.variable_scope('yolo'):
        #使用darknet53进行特征提取后在进行分类，部分输出进行上采样，最终输出作为yolo结果
            conv2d_57,conv2d_59, conv_index = self._yolo_block(conv,512,num_anchors*(num_classes+5),conv_index=conv_index,training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            #此处即开始第二个特征层的获取，进行上采样之前先卷积加归一化
            conv2d_60 = self._conv2d_layer(conv2d_57,filter_num=256,kernel_size=1,strides=1,name="conv2d_"+str(conv_index))
            conv2d_60 = self._batch_normalization(conv2d_60,name='batch_normalization_'+str(conv_index),norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon,training=training)
            conv_index += 1
            #此处进行上采样，并和上一尺寸数据concat
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60,[2*tf.shape(conv2d_60)[1],2*tf.shape(conv2d_60)[1]],name='upSampel_0')
            route0 = tf.concat([upSample_0,conv2d_43],axis=-1,name='route_0')

            conv2d_65,conv2d_67,conv_index = self._yolo_block(route0,256,num_anchors*(num_classes+5),conv_index=conv_index,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            #第三特征层获取，同上
            conv2d_68 = self._conv2d_layer(conv2d_65,filter_num=128,kernel_size=1,strides=1,name='conv2d_'+str(conv_index))
            conv2d_68 = self._batch_normalization(conv2d_68,name='batch_normalization_'+str(conv_index),norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index += 1

            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,[2*tf.shape(conv2d_68)[1],2*tf.shape(conv2d_68)[1]],name='upSample_1')

            route1 = tf.concat([upSample_1,conv2d_26],axis=-1,name='route_1')

            _,conv2d_75,_ = self._yolo_block(route1,128,num_anchors*(num_classes+5),conv_index=conv_index,training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)

            return [conv2d_59,conv2d_67,conv2d_75]