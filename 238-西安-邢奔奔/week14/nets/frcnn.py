from nets.resnet import resnet50
from keras.layers import Input,Dense,Conv2D,Reshape,TimeDistributed,Flatten
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv
from nets.resnet import classifier_layer
from nets.frcnn_training import Generator
def get_rpn(base_layers,num_anchors):
    """
        rpn网络，获取分类信息和anchors信息
    :param base_layers:fetarue_map作为基本输入
    :param num_anchors:生成的anchors数量
    :return: rpn网络输出
    """
    x = Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='normal',name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors,(1,1),activation='sigmoid',kernel_initializer='uniform',name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors*4,(1,1),activation='linear',kernel_initializer='zero',name='rpn_out_regress')(x)

    x_class = Reshape((-1,1),name='classification')(x_class)
    x_regr = Reshape((-1,4),name='regression')(x_regr)
    return [x_class,x_regr,base_layers]
def get_classifier(base_layers,input_rois,num_rois,nb_classes=21,trainable=False):
    pooling_regions = 14
    input_shape = (num_rois,14,14,1024)
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])
    out = classifier_layer(out_roi_pool,input_shape=input_shape,trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes,activation='softmax',kernel_initializer='zero'),name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4*(nb_classes-1),activation='linear',kernel_initializer='zero'),name='dense_regress_{}'.format(nb_classes))(out)
    return [out_class, out_regr]

def get_model(config,num_classes):
    inputs = Input(shape=(None,None,3))
    roi_input = Input(shape=(None,4))
    base_layers = resnet50(inputs)
    #计算anchors数量
    num_anchors = len(config.anchors_box_scale)*len(config.anchor_box_ratios)
    rpn = get_rpn(base_layers,num_anchors)
    #将inputs和rpn输出的anchors做映射
    model_rpn = Model(inputs,rpn[:2])
    #获取分类网络，此处一直报错
    classifier = get_classifier(base_layers,roi_input,config.num_rois,nb_classes=num_classes,trainable=True)
    model_classifier = Model([inputs,roi_input],classifier)

    model_all = Model([inputs, roi_input],rpn[:2]+classifier)
    return model_rpn, model_classifier, model_all


def get_predict_model(config,num_classes):
    """
        获取预测模型
    :param config:
    :param num_classes:
    :return: model_rpn, model_classifier
    """
    inputs = Input(shape=(None,None,3))
    roi_input = Input(shape=(None,4))
    #feature_map为特征提取层网络的输出，就是base_layers
    feature_map_input = Input(shape=(None,None,1024))

    base_layers = resnet50(inputs)
    num_anchors = len(config.anchors_box_scale) * len(config.anchor_box_ratios)

    rpn = get_rpn(base_layers,num_anchors)
    model_rpn = Model(inputs,rpn)

    classifier = get_classifier(feature_map_input,roi_input,config.num_rois,nb_classes=num_classes,trainable=True)
    model_classifier_only = Model([feature_map_input,roi_input],classifier)
    return model_rpn, model_classifier_only
