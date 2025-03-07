import numpy as np
import tensorflow as tf
from PIL import Image


def load_weights(var_list,weight_file):



    with open(weight_file,"rb") as fp:
        _ = np.fromfile(fp,dtype=np.int32,count=5)
        weights = np.fromfile(fp,dtype=np.float32)

    ptr , i = 0, 0
    assign_opt = []
    while i < len(var_list)-1:
        var1 = var_list[i]
        var2 = var_list[i+1]
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                gamma,beta,mean,var = var_list[i+1:i+5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr+num_params].reshape(shape)
                    ptr += num_params
                    assign_opt.append(tf.assign(var,var_weights,validate_shape=True))
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr+bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_opt.append(tf.assign(bias,bias_weights,validate_shape=True))
                i += 1

        shape = var1.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr+num_params].reshape(shape[3],shape[2],shape[0],shape[1])
        var_weights = np.transpose(var_weights,(2,3,1,0))
        ptr += num_params
        assign_opt.append(tf.assign(var1, var_weights, validate_shape=True))
        i += 1

    return assign_opt

def letterbox_image(image, size):

    image_w, image_h = image.size
    w, h = size
    #按照长宽比进行缩放
    new_w = int(image_w*min(w*1.0/image_w,h*1.0/image_h))
    new_h = int(image_h*min(w*1.0/image_w,h*1.0/image_h))
    resized_image = image.resize((new_w,new_h),Image.BICUBIC)

    boxed_image = Image.new('RGB',size,(128,128,128))
    boxed_image.paste(resized_image,((w-new_w)//2,(h-new_h)//2))

    return boxed_image

def draw_box(image,bbox):

    xmin,ymin,xmax,ymax,label = tf.split(value=bbox,num_or_size_splits=5,axis=2)
    height = tf.cast(tf.shape(image)[1],tf.float32)
    weight = tf.cast(tf.shape(image)[2],tf.float32)

    new_bbox = tf.concat([tf.cast(ymin,tf.float32)/height,tf.cast(xmin,tf.float32)/weight,tf.cast(ymax,tf.float32)/height,tf.cast(xmax,tf.float32)/weight],2)
    new_image = tf.image.draw_bounding_boxes(image,new_bbox)
    tf.summary.image('input',new_image)

