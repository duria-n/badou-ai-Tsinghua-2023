

class Config():
    def __init__(self):
        self.anchors_box_scale = [128,256,512]
        self.anchor_box_ratios = [[1,1],[1,2],[2,1]]
        self.rpn_stride = 16
        self.num_rois = 32
        self.verbose = True
        self.model_path='.\logs\model.h5'
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        self.class_min_overlap = 0.1
        self.class_max_overlap = 0.5
        self.class_regr_std = [8.0,8.0,4.0,4.0]
