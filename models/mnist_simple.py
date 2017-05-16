


import tensorflow as tf



import sys



from util import weight_variable, bias_variable, conv2d, max_pool_2x2



def make_model(x,y_):
    
    #1st conv
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

    # 1st max pool
    h_pool1 = max_pool_2x2(h_conv1)

    #second conv
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # second max pool
    h_pool2 = max_pool_2x2(h_conv2)


    #fc
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    
    # pre softmax fc
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    #loss: softmax followed by cross_entropy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
    
    #accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy



def test():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    make_model(x,y_)



if __name__ == "__main__":
    test()








