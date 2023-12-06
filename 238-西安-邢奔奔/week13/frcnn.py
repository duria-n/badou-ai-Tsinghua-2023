import os
import cv2
import numpy as np
import colorsys
import pickle
import nets.frcnn as frcnn
import keras
from nets.frcnn_training import get_new_img_size
from keras.layers import Input
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image,ImageFont,ImageDraw
from utils.utils import BBoxUtility
from utils.anchors import get_anchor
from utils.config import Config
import copy
import math


class FRCNN():
    _defaults= {
        "model_path":'model_data/voc_weights.h5',
        "class_path":'model_data/voc_classes.txt',
        "confidence":0.7,
    }