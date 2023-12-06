
import keras
import numpy as np
import tensorflow as tf
from utils.Config import Config
import matplotlib.pyplot as plt
config = Config()
import keras.backend as K

def generate_anchors(size=None,ratios=None):
    """
        这里是生成anchors列表
    :param size:
    :param ratios:
    :return:
    """
    if size is None:
        size = config.anchors_box_scale
    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(ratios) * len(size)
    anchors = np.zeros((num_anchors,4))
    anchors[:,2:] = np.tile(size,(2,len(ratios))).T
    for i in range(len(ratios)):
        anchors[3*i:3*i+3,2] = anchors[3*i:3*i+3,2] * ratios[i][0]
        anchors[3*i:3*i+3,3] = anchors[3*i:3*i+3,3] * ratios[i][1]
    #更新锚框的中心点位置，将锚框位置从左上角移到中心
    anchors[:,0::2] -= np.tile(anchors[:,2]*0.5,(2,1)).T
    anchors[:,1::2] -= np.tile(anchors[:,3]*0.5,(2,1)).T
    return anchors

def shift(shape,anchors,strides=config.rpn_stride):
    """

    :param shape:
    :param anchors:
    :param strides:
    :return: 每一行代表一个平移后的锚点框
    """
    #计算水平和垂直方向的偏移量，基于图片尺寸和步长
    shift_x = (np.arange(0,shape[0],dtype=K.floatx()) + 0.5) * strides
    shift_y = (np.arange(0,shape[1],dtype=K.floatx()) + 0.5) * strides
    #使用np.meshgrid生成网格上的所有节点，对应网格中每个位置
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    #重塑每个点的坐标将其从二维数组转为一位数组，以便后续使用
    shift_x = np.reshape(shift_x,[-1])
    shift_y = np.reshape(shift_y,[-1])

    shifts = np.stack([shift_x,shift_y,shift_x,shift_y],axis=0)
    shifts = np.transpose(shifts)
    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]
    #将偏移量和锚点坐标相加，得到调整后的坐标
    shifted_anchors = np.reshape(anchors,[1,number_of_anchors,4])+np.array(np.reshape(shifts,[k,1,4]),K.floatx())
    shifted_anchors = np.reshape(shifted_anchors,[k*number_of_anchors,4])
    return shifted_anchors

def get_anchors(shape,width,height):
    anchors = generate_anchors()
    #对平移后的锚点框做归一化
    network_anchors = shift(shape,anchors)
    network_anchors[:,0] = network_anchors[:,0] / width
    network_anchors[:,1] = network_anchors[:,1] / height
    network_anchors[:,2] = network_anchors[:,2] / width
    network_anchors[:,3] = network_anchors[:,3] / height
    #这里使用np.clip是一种保险措施，可能存在计算误差，或者anchors在窗口之外，较大的滑动窗口和anchors尺寸
    network_anchors = np.clip(network_anchors,0,1)
    return network_anchors

