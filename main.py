
# coding: utf-8

# In[1]:

import tensorflow as tf
import sys


# In[2]:

from models.resnet101 import make_model
from get_data.dummy_imagenet import load_tr_set, get_shapes
from slurm_tf_helper.setup_clusters import setup_slurm_cluster
import os
import numpy as np
import time


# In[3]:

def main(_):
    batch_size = 128
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)
    
    if job_name == "ps":
        print "time started: ", time.time()
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)):
            
            x_shape, y_shape = get_shapes()
            x,y, loss = make_model(x_shape,batch_size)
            
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(0.01)
            train_op = opt.minimize(loss, global_step=global_step)

        
        
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        with tf.train.MonitoredTrainingSession(is_chief=(task_index == 0),
                                 master=server.target,
                                 checkpoint_dir="./logs",
                                 hooks=hooks) as mon_sess:
            ims,lbls = load_tr_set(task_index, num_tasks)
            num_ex = lbls.shape[0]
            step = 0
            while not mon_sess.should_stop():
                start = (step * batch_size) % num_ex
                stop = (start + batch_size) % num_ex
                slice_ = slice(start, stop)
                batch = (ims[slice_], lbls[slice_])
                _, step, loss_ = mon_sess.run([train_op, global_step, loss],feed_dict={x:batch[0], y: batch[1]})
                print "loss for task id %i is %8.2f" % (task_index, loss_)


# In[4]:

if __name__ == "__main__":
    tf.app.run(main=main)


# In[1]:

#! jupyter nbconvert --to script ./main.ipynb


# In[ ]:



