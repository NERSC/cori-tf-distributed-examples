


import tensorflow as tf



tf.__version__



import tensorflow.contrib.keras as keras

from keras.layers import Conv2D, BatchNormalization, Activation, Input
from util import conv_block, identity_block



def make_model(x_shape,y_shape):
    layers = {}
    img_input = Input(shape=inp_shape)
    bn_axis = 3

    layers["input"] = img_input

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid',name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    layers["conv1"] = x

    x = MaxPooling2D((3, 3), strides=(2, 2),padding="valid")(x)
    
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    layers["conv2_x"] = x


    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2,2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    layers["conv3_x"] = x

    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',strides=(2,2))
    block = 'b'
    for i in range(22):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=block)
        #increment letter
        block = chr(ord(block) + 1)

    layers["conv4_x"] = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(2,2))
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    layers["conv5_x"] = x
    layers["last"] = x
    return layers

