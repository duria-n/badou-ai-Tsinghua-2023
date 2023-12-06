import numpy as np
import tensorflow as tf


class BBoxUtility(object):

    def __init__(self, priors=None, overlap_threshold=0.7, ignore_threshold=0.3,
                 num_thres=0.7, top_k=300):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self.nms_threshold = num_thres
        self.top_k = top_k
        self.boxes = tf.placeholder(dtype='float32', shape=(None, 4))
        self.scores = tf.placeholder(dtype='float32', shape=(None,))
        self.nms = tf.image.non_max_suppression(boxes=self.boxes, scores=self.scores,
                                                max_output_size=self.top_k, iou_threshold=self.nms_threshold)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

    def nms_thres(self):
        return self.num_threshold

    def nms_thres_setter(self, value):
        self.nms_threshold = value
        self.nms = tf.image.non_max_suppression(boxes=self.boxes, scores=self.scores,
                                                max_output_size=self.top_k,
                                                iou_threshold=self.nms_threshold)

    def top_k(self):
        return self.top_k

    def top_k_setter(self, value):
        self.top_k = value
        self.nms = tf.image.non_max_suppression(boxes=self.boxes, scores=self.scores,
                                                max_output_size=self.top_k,
                                                iou_threshold=self.nms_threshold)

    def iou(self, box):
        inter_uplef = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.maximum(self.priors[:2:4], box[2:])

        inter_wh = inter_botright - inter_uplef
        # 这里存疑，右下角减去左上角是什么意思
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        area_priors = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])

        area_union = area_priors + area_true - inter

        iou = inter / area_union
        return iou

    