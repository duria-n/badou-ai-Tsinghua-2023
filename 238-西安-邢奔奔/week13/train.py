from utils.config import Config
from utils.utils import BBoxUtility





if __name__ == '__main__':
    config = Config()
    num_classes = 21
    EPOCH = 100
    EPOCH_LENGTH = 2000
    bbox_utils = BBoxUtility(overlap_threshold=config.rpn_max_overlap,ignore_threshold=config.rpn_min_overlap)
    annotation_path = '2007_trian.txt'