import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw

import config
from utils import letterbox_image, load_weights
from yolopredict import yolo_predictor

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

def detect(image_path,model_path,yolo_weights = None):
    '''
        加载模型，进行预测
    :param image_path: 图片路径
    :param model_path: 模型路径，无权重路径时使用
    :param yolo_weights: 权重路径
    :return:
    '''

    #图像预处理
    image = Image.open(image_path)
    resize_image = letterbox_image(image,(416,416))
    image_data = np.array(resize_image,dtype=np.float32)
    image_data /= 255
    #将hwc转换为nhwc
    image_data = np.expand_dims(image_data,axis=0)

    input_image_shape = tf.placeholder(dtype=tf.int32,shape=(2,))
    input_image = tf.placeholder(shape=[None,416,416,3],dtype=tf.float32)
    #使用yolo_predictor进行预测，
    predictor = yolo_predictor(config.obj_threshold,config.nms_threshold,config.classes_path,config.anchors_path)

    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image,input_image_shape)

                load_op = load_weights(tf.global_variables(scope='predict'),weight_file=yolo_weights)
                sess.run(load_op)

                #进行预测
                out_boxes, out_scores, out_classes = sess.run([boxes,scores,classes],
                                                              feed_dict={input_image:image_data,
                                                                    input_image_shape:[image.size[1],image.size[0]]})
        else:
            #进行训练
            boxes, scores, classes = predictor.predict(input_image,input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess,model_path)
            out_boxes, out_scores, out_classes = sess.run([boxes,scores,classes],
                                                          feed_dict={input_image:image_data,input_image_shape:[image.size[1],image.size[0]]})
        #找到的box总数
        print('Found {} boxes for {}'.format(len(out_boxes),'img'))
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
        #边框绘制次数
        thickness = (image.size[0]+image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class= predictor.class_names[c]
            box = out_boxes[i]
            scores = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class,scores)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, buttom, right = box
            top = max(0,np.floor(top+0.5).astype('int32'))
            left = max(0,np.floor(left+0.5).astype('int32'))
            buttom = min(image.size[1]-1,np.floor(buttom+0.5).astype('int32'))
            right = min(image.size[0]-1,np.floor(right+0.5).astype('int32'))
            print(label,(left,top),(right,buttom))
            print(label_size)
            #判断矩形框上方是否有空间绘制标签，如果有则将标签位置设置在据矩形框上方，否则在矩形框下方设置标签位置
            if top - label_size[1] >= 0:
                text_origin = np.array([left,top-label_size[1]])
            else:
                text_origin = np.array([left,top+1])

            for i in range(thickness):
                draw.rectangle(
                    [left+i,top+i,right-i,buttom-i],
                    outline=predictor.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=predictor.colors[c])
                draw.text(text_origin,label,fill=(0,0,0),font=font)
            del draw
        image.show()
        image.save('./img/result1.jpg')


if __name__ == '__main__':

    if config.pre_train_yolo3 == True:
        detect(config.image_file,config.model_dir,config.yolo3_weights_path)
    else:
        detect(config.image_file,config.model_dir)