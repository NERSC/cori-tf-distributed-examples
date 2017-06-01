
# coding: utf-8

# In[3]:

import tensorflow as tf
import tensorflow.contrib.keras as keras


# In[12]:

from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, AveragePooling2D, Flatten, Dense
from util import conv_block, identity_block
from keras import backend as K



# In[22]:

num_classes = 1000


# In[60]:

def make_model(x_shape, batch_size):
    K.set_learning_phase(1)
    y = tf.placeholder(dtype=tf.int32,shape=(batch_size,))# Input(dtype=tf.int32, shape=y_shape)

    img_input = Input(batch_shape= tuple( [batch_size] + list(x_shape)))
    bn_axis = 3


    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid',name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)


    x = MaxPooling2D((3, 3), strides=(2, 2),padding="valid")(x)
    
    
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')



    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2,2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')


    
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',strides=(2,2))
    block = 'b'
    for i in range(22):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=block)
        #increment letter
        block = chr(ord(block) + 1)



    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=(2,2))
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = AveragePooling2D((7, 7), name='avg_pool')(x)


    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x,labels=y))
    
    return img_input, y, loss


# In[2]:

#! jupyter nbconvert --to script resnet101.ipynb


# In[ ]:



