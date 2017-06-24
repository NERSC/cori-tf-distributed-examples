
# coding: utf-8

# In[1]:

import tensorflow as tf
import sys


# In[2]:

from models.resnet_cifar import make_model
from configs import get_configs
from slurm_tf_helper.setup_clusters import setup_slurm_cluster
import os
import numpy as np
import time
import importlib


# In[5]:

configs = get_configs()

data_module = importlib.import_module("get_data." + configs["dataset"])

get_generator, get_num_classes = data_module.get_generator, data_module.get_num_classes


# In[3]:

def main(_):
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)
    
    if job_name == "ps":
        print "time started: ", time.time()
        server.join()
    
    elif job_name == "worker":
        generator = get_generator(num_tasks, task_id=task_index, batch_size=configs["batch_size"], path_to_h5file=configs["path_to_h5_file"])
        num_classes = get_num_classes()
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)):
            
            #set up model variables/ops
            x,y, loss = make_model(generator.im_shape[1:], configs["batch_size"], num_classes)
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(0.01)
            
            if configs["mode"] == "sync":
                opt = tf.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_tasks,
                               total_num_replicas=num_tasks)
            train_op = opt.minimize(loss, global_step=global_step)

        

        
        
        #set up training ops
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        
        if configs["mode"] == "sync":
            hooks.append(opt.make_session_run_hook(is_chief=(task_index == 0)))
            
        
        with tf.train.MonitoredTrainingSession(is_chief=(task_index == 0),
                                 master=server.target,
                                 checkpoint_dir="./logs",
                                 hooks=hooks) as mon_sess:
            
            
            while not mon_sess.should_stop():
                
                for ims,lbls in generator:
                    iteration_start = time.time()
                    _, step, loss_ = mon_sess.run([train_op, global_step, loss],feed_dict={x:ims, y: lbls})
                    print step
                    print "loss for task id %i is: %8.4f " % (task_index, loss_)
                    print "time for iteration for task %i, batch size %i is %5.2f" % (task_index, configs["batch_size"], time.time() - iteration_start)
                


# In[1]:

if __name__ == "__main__":
    tf.app.run(main=main)


# In[ ]:

get_ipython().system(u' jupyter nbconvert --to script ./main.ipynb')


# In[ ]:



