import numpy as np
import keras
import tensorflow as tf
from config import Config
import matplotlib.pyplot as plt

config = Config()

def generate_anchors(size=None,ratios=None):
    if size is None:
        size = config.anchor_box_scale
    if ratios is None:
        ratios = config.anchor_box_ratios
    num_anchors = len(size) * len(ratios)

    anchors = np.zeros((num_anchors,4))
    anchors[:,2:] = np.tile(size,(2,len(ratios))).T
    # print(anchors)
    # print('-------------------------------------------')
    for i in range(len(ratios)):
        anchors[3*i:3*i+3,2] = anchors[3*i:3*i+3,2]*ratios[i][0]
        anchors[3*i:3*i+3,3] = anchors[3*i:3*i+3,3]*ratios[i][1]
    # print(anchors) 这里实现9个位置的anchors按比例缩放
    return anchors

def shift(shape,anchors,strides=config.rpn_strides):
    shift_x = (np.arrange(0,shape[0],dtype=keras.backend.floatx())+0.5)*strides
    shift_y = (np.arrange(0,shape[1],dtype=keras.backend.floatx())+0.5)*strides

if __name__ == '__main__':
    anchors = generate_anchors()
