
# coding: utf-8

# In[3]:

import tensorflow as tf
import os


# In[4]:

def weight_variable(shape):
    #add a tesnor initialization to the com graph
    initial = tf.truncated_normal(shape, stddev=0.1)
    #make this tensor a variable and return it
    return tf.Variable(initial)

def bias_variable(shape):
    #intiialize all biases at 0.1
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    # add 2d conv op to the graph that convolves x with a filter W
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')



def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# In[ ]:

def expand_nodelist(node_string):
    pref, suff  = node_string.split('[')

    suff = suff. split(']')[0].split(',')
    nodes =[]
    for s in suff:
        if '-' not in s:
            nodes.append("%s%s" % (pref, s))
            continue
        beg,end = s.split('-')
        num_len=len(beg)
        for id in range(int(beg),int(end) + 1):
            j= "%s%0" + str(num_len) + "d"
            nodes.append(j % (pref, id))
    return nodes

if __name__ == "__main__":
    node_string = "nid22[8,11-99,101-120]"
    nodes = expand_nodelist(node_string)
    print nodes

