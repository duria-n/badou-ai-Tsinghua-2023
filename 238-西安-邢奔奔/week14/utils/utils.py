import numpy as np
import tensorflow as tf
import keras
import tensorflow as tf
from PIL import Image
import math

class BBoxUtility(object):
    def __init__(self,priors=None,overlap_threshold=0.7,ignore_threshold=0.3,
                 nms_thresh = 0.7,top_k=300):
        #先验框默认为None
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k
        #每个框有4个坐标值，得分有一个值
        self.boxes = tf.placeholder(dtype='float32',shape=(None,4))
        self.scores = tf.placeholder(dtype='float32',shape=(None,))
        self.nms = tf.image.non_max_suppression(self.boxes,self.scores,
                                                self._top_k,iou_threshold=self._nms_thresh)
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU':0}))


    def nms_thresh(self):
        return self._nms_thresh

    def nms_thresh(self,value):
        self._nms_thresh = value
        self.nms = tf.image.non_max_suppression(self.boxes,self.scores,
                                                self._top_k,iou_threshold=self._nms_thresh)

    def top_k(self):
        return self._top_k

    def top_k(self,value):
        self._top_k = value
        self.nms = tf.image.non_max_suppression(self.boxes,self.scores,
                                                self._top_k,iou_threshold=self._nms_thresh)

    def iou(self,box):
        inter_upleft = np.maximum(self.priors[:,:2],box[:2])
        inter_botright = np.minimum(self.priors[:,2:4],box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh,0)
        inter = inter_wh[:,0]*inter_wh[:,1]

        area_true = (box[2] - box[0]) *(box[3]-box[1])
        area_gt = (self.priors[:,2]-self.priors[:,0]) * (self.priors[:,3]-self.priors[:,1])

        union = area_true+area_gt - inter

        iou = inter / union

        return iou

    def encode_box(self,box,return_iou=True):
        iou = self.iou(box)
        #这里的4+1是因为有4个坐标值，1是因为有可选的iou值，至于是否返回由参数return_iou决定，这里encoded_box储存的数据是xywh
        encoded_box = np.zeros((self.num_priors,4+return_iou))

        assign_mask = iou > self.overlap_threshold
        #这里为了确保assign_mask只要有一个元素，如果为空，则返回iou值最大的标记
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        #这里是如果需要返回iou，就将编码信息的最后一列以外的数值替换为对应的iou值
        if return_iou:
            encoded_box[:,-1][assign_mask] = iou[assign_mask]
        #储存与目标框匹配的先验框
        assigned_priors = self.priors[assign_mask]
        #计算目标框的中心位置和宽高
        box_center = 0.5 * (box[:2]+box[2:])
        box_wh = box[2:]-box[:2]
        #计算先验框的中心位置和宽高
        assigned_priors_center = 0.5 * (assigned_priors[:,:2]+assigned_priors[:,2:4])
        assigned_priors_wh = (assigned_priors[:,2:4] - assigned_priors[:,:2])
        #首先计算目标框和先验框的中心差值，然后做归一化，确保编码的尺度独立于先验框的尺寸，再进行坐标缩放
        encoded_box[:,:2][assign_mask] = box_center-assigned_priors_center
        encoded_box[:,:2][assign_mask] /=assigned_priors_wh
        encoded_box[:,:2][assign_mask] *= 4
        #这里是对宽高信息进行尺度缩放
        encoded_box[:,2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:,2:4][assign_mask] *= 4
        #返回压缩为一维数组的编码信息
        return encoded_box.ravel()

    def ignore_box(self,box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors,1))

        assign_mask = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        ignored_box[:,0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    #为输入的box分配对应的先验框，将要忽略的和与真是框匹配的先验框使用矩阵来表现
    def assign_boxes(self,boxes,anchors):
        #获取先验框的数量和先验框
        self.num_priors = len(anchors)
        self.priors = anchors
        #创建分配矩阵，尺寸为（self.num_priors，5）最后一个维度是是否忽略的标签
        assignment = np.zeros((self.num_priors,4+1))
        #将最后一列的元素设置为0，表示初始分配状态
        assignment[:,4] = 0.0
        #如果输入box为空，则直接返回空的分配矩阵
        if len(boxes) == 0:
            return assignment

        #self.ignore_box,将boxes的前四列按照第一轴应用与self.ignore_box函数，生成ignore_boxes
        ignored_boxes = np.apply_along_axis(self.ignore_box,1,boxes[:,:4])
        #-1为占位符，第一维度为边界框数量
        ignored_boxes = ignored_boxes.reshape(-1,self.num_priors,1)
        #ignored_boxes[:,:,0]是取ignored_boxes的第一个元素，中的所有行和所有列的第一个元素（索引0）进行选择，按行方向计算最大值，即取每列的最大值
        # 第三个维度包含了框的不同属性，而前两个维度对应于不同的框和参考框
        ignore_iou = ignored_boxes[:,:,0].max(axis=0)
        #指示哪些先验框与任何边界框具有非零的 IoU，将满足 ignored_iou_mask 条件的先验框的分配状态设置为 -1，表示这些先验框应该被忽略
        ignore_iou_mask = ignore_iou > 0
        assignment[:,4][ignore_iou_mask] = -1


        encoded_boxes = np.apply_along_axis(self.encode_box,1,boxes[:,:4])

        encoded_boxes = encoded_boxes.reshape(-1,self.num_priors,5)


        best_iou = encoded_boxes[:,:,-1].max(axis=0)

        best_iou_idx = encoded_boxes[:,:,-1].argmax(axis=0)

        best_iou_mask = best_iou > 0

        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)

        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        #表示这些先验框与某些边界框相匹配
        assignment[:,4][best_iou_mask] = 1
        #返回最终的匹配矩阵
        return assignment


    #将预测的信息进行解码，解码过程和编码过程相比刚好反过来，这里是已知编码信息进行解码
    def decode_boxes(self,mbox_loc,mbox_priorbox):
        #计算先验框宽高
        prior_width = mbox_priorbox[:,2] - mbox_priorbox[:,0]
        prior_height = mbox_priorbox[:,3] - mbox_priorbox[:,1]
        #先验框中心点
        prior_center_x = 0.5 * (mbox_priorbox[:,2]+mbox_priorbox[:,0])
        prior_center_y = 0.5 * (mbox_priorbox[:,3]+mbox_priorbox[:,1])

        #计算真实框相对于先验框中心的偏移量，然后得到预测框坐标
        decode_bbox_center_x = mbox_loc[:,0] * prior_width / 4
        decode_bbox_center_x +=prior_center_x
        decode_bbox_center_y = mbox_loc[:,1] * prior_height / 4
        decode_bbox_center_y +=prior_center_y

        #计算预测框的宽高
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height

        #获得预测框的左上角和右下角坐标哦
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        #将四个坐标扩展为一个数组，将decode_bbox_xmin转为二维数组，从而成为一个列向量
        decode_bbox = np.concatenate((decode_bbox_xmin[:,None],decode_bbox_ymin[:,None],
                                     decode_bbox_xmax[:,None],decode_bbox_ymax[:,None]),axis=-1)

        decode_bbox = np.minimum(np.maximum(decode_bbox,0.0),1.0)
        return decode_bbox

    def detection_out(self,predictions,mbox_priorbox,num_classes,keep_top_k=300,confidence_threshold=0.5):
        """

        :param predictions: 0:class 1:box_regr
        :param mbox_priorbox:
        :param num_classes:
        :param keep_top_k:
        :param confidence_threshold:
        :return:
        """
        #取出类别置信区间
        mbox_conf = predictions[0]
        #取出框的位置
        mbox_loc = predictions[1]
        #取出anchors位置
        mbox_priorbox = mbox_priorbox
        results = []

        for i in range(len(mbox_loc)):
            #遍历所有尺度的位置信息
            results.append([])
            #对当前尺度和anchors的位置信息进行解码，得到预测边界框坐标
            decode_bbox = self.decode_boxes(mbox_loc[i],mbox_priorbox)
            for c in range(num_classes):
                #取出当前尺度，当前类别的置信度
                c_confs = mbox_conf[i,:,c]
                #创建置信度大于阈值的掩码
                c_confs_m = c_confs > confidence_threshold
                #如果存在大于阈值的预测
                if len(c_confs[c_confs_m]) > 0:
                    #取出得分高于阈值的预测框的坐标和置信度
                    boxes_to_process = decode_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]

                    #对边界框进行筛选，保留边界框的索引
                    feed_dict = {self.boxes:boxes_to_process, self.scores:confs_to_process}
                    idx = self.sess.run(self.nms,feed_dict=feed_dict)
                    #根据非极大值抑制的结果提取可以保留的anchors和对情的置信度
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:,None]
                    #创建改类别对应的标签
                    labels = c * np.ones((len(idx),1))
                    #将标签，置信度和坐标信息连接，合成一个预测结果
                    c_pred = np.concatenate((labels,confs,good_boxes),axis=1)
                    #将预测结果扔进事先准备的容器
                    results[-1].extend(c_pred)
            #如果当前存在保留的预测信息
            if len(results[-1])>0:
                #将预测信息转为数组形式
                results[-1] = np.array(results[-1])
                #取出当前预测结果，并按照置信度进行从大到小的排序
                argsort = np.argsort(results[-1][:,1])[::-1]
                results[-1] = results[-1][argsort]
                #保留topk的数据
                results[-1] = results[-1][:keep_top_k]

        return results

    def nms_for_out(self,all_labels,all_confs,all_bboxes,num_classes,nms):
        results = []
        nms_out = tf.image.non_max_suppression(self.boxes,self.scores,self._top_k,iou_threshold=nms)

        for c in range(num_classes):
            c_pred = []
            mask = all_labels ==c
            if len(all_confs[mask]) > 0:
                boxes_to_process = all_bboxes[mask]
                confs_to_process = all_confs[mask]

                feed_dict = {self.boxes:boxes_to_process,self.scores:confs_to_process}
                idx = self.sess.run(nms_out,feed_dict=feed_dict)

                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]

                labels = c * np.ones((len(idx),1))
                c_pred = np.concatenate((labels,confs,good_boxes),axis=1)
            results.extend(c_pred)
        return results

#
# if __name__ == '__main__':
#     priors = np.array([[1,7,4,3]])
#     bb = BBoxUtility(priors=priors)
#     bb.iou([2,6,5,2])























