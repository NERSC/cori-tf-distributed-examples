
# coding: utf-8

# In[4]:

import tensorflow as tf
import sys


# In[2]:

from models.resnet101 import make_model
from configs import get_configs
from slurm_tf_helper.setup_clusters import setup_slurm_cluster
import os
import numpy as np
import time
import importlib


# In[4]:

configs = get_configs()

data_module = importlib.import_module("get_data." + configs["dataset"])

load_tr_set, get_shapes = data_module.load_tr_set, data_module.get_shapes


# In[3]:

def main(_):
    batch_size = 128
    
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)
    task_indices_for_timing = [0]
    if job_name == "ps":
        print "time started: ", time.time()
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)):
            #set up model variables/ops
            x_shape, y_shape = get_shapes()
            x,y, loss = make_model(x_shape, batch_size)
            
    
            
            global_step = tf.contrib.framework.get_or_create_global_step()
            opt = tf.train.AdamOptimizer(0.01)
            train_op = opt.minimize(loss, global_step=global_step)

        
        
        

        # set up timing ops
        iteration_time = tf.placeholder(dtype=tf.float64,)
        tf.summary.scalar("iter_time", iteration_time)
        merged = tf.summary.merge_all()
        
        #set up training ops
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        with tf.train.MonitoredTrainingSession(is_chief=(task_index == 0),
                                 master=server.target,
                                 checkpoint_dir="./logs",
                                 hooks=hooks) as mon_sess:
            
            # only save timings for one of the tasks
            if task_index in task_indices_for_timing:
                train_writer = tf.summary.FileWriter(os.path.join(configs["logdir"],configs["exp_name"],str(num_tasks) + "_nodes","task_%i"%task_index), mon_sess.graph)
            
            
            #get chunk of data
            ims,lbls, total_images = load_tr_set(task_index, num_tasks)
            
            steps_per_epoch = int(total_images / float(batch_size))
            num_ex = lbls.shape[0]
            step = 0
            iter_times = []
            avg_iter_time = 0
            while not mon_sess.should_stop():
                iteration_start = time.time()
                
                #get batch out of the chunk
                start = (step * batch_size) % num_ex
                stop = (start + batch_size) % num_ex
                print start,stop, task_index
                slice_ = slice(start, stop)
                batch = (ims[slice_], lbls[slice_])
                
                #run one iteration of training
                _, step, loss_, summary = mon_sess.run([train_op, global_step, loss, merged],feed_dict={x:batch[0], 
                                                                                                        y: batch[1],
                                                                                                       iteration_time: avg_iter_time})
                
                print "loss for task id %i is: " % (task_index)
                print loss_
                
                # update average iteration time
                iteration_end = time.time()
                iter_times.append(iteration_end - iteration_start)
                avg_iter_time = sum(iter_times) / float(len(iter_times))
                print "running average iter_time for task %i is %8.4f" % (task_index, avg_iter_time)
                
                # update timing for this task index for this concurrency
                if task_index in task_indices_for_timing:
                    num_nodes = num_tasks
                    train_writer.add_summary(summary, num_nodes)
                    train_writer.close()
                    train_writer.reopen()
                


# In[1]:

if __name__ == "__main__":
    tf.app.run(main=main)


# In[7]:

#! jupyter nbconvert --to script ./main.ipynb


# In[ ]:



