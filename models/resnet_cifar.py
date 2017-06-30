# thanks to F. Chollet for this one:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, AveragePooling2D, Flatten, Dense

from util import conv_block, identity_block
import keras.backend as K


def make_model(x_shape, batch_size=128, num_classes=10):
    y = tf.placeholder(dtype=tf.int32,shape=(batch_size,))# Input(dtype=tf.int32, shape=y_shape)
    K.set_learning_phase(1)
    img_input = Input(batch_shape= tuple( [batch_size] + list(x_shape)))
    bn_axis = 3


    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)

   
    x = conv_block(x, 3, [16,16,16], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [16,16,16], stage=2, block='b')
    x = identity_block(x, 3, [16,16,16], stage=2, block='c')
    x = identity_block(x, 3, [16,16,16], stage=3, block='d')

    
    x = conv_block(x, 3, [32,32,32], stage=3, block='a', strides=(2,2))
    x = identity_block(x, 3, [32,32,32], stage=3, block='b')
    x = identity_block(x, 3, [32,32,32], stage=3, block='c')
    x = identity_block(x, 3, [32,32,32], stage=3, block='d')

    
    
    x = conv_block(x, 3, [64,64,64], stage=3, block='a', strides=(2,2))
    x = identity_block(x, 3, [64,64,64], stage=3, block='b')
    x = identity_block(x, 3, [64,64,64], stage=3, block='c')
    x = identity_block(x, 3, [64,64,64], stage=3, block='d')

    
    x = AveragePooling2D((8, 8), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x,labels=y))
    
    return img_input, y, loss

