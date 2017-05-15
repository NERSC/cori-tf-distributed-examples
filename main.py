
# coding: utf-8

# In[1]:

import tensorflow as tf
import sys


# In[2]:

# from nbfinder import NotebookFinder
# sys.meta_path.append(NotebookFinder())
from model import make_model
from get_data import load_tr_set
from setup_clusters import setup_cluster
import os
import numpy as np
import time


# In[3]:

def main(_):
    batch_size = 128
    cluster_type = sys.argv[1]
    cluster, server, task_index, num_tasks, job_name = setup_cluster(num_ps=1)
    
    if job_name == "ps":
        print "time started: ", time.time()
        server.join()
    elif job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)):
            x = tf.placeholder(tf.float32, shape=[None, 28, 28,1])
            y_ = tf.placeholder(tf.float32, shape=[None, 10])
            # will put all variables on ps nodes
            loss, accuracy = make_model(x, y_)
            global_step = tf.Variable(0)
            opt = tf.train.AdagradOptimizer(0.01)
            train_op = opt.minimize(loss, global_step=global_step)
            init_op = tf.initialize_all_variables()
        is_chief=(task_index == 0)
        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir="./logs",
                                 init_op=init_op,
                                 global_step=global_step)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        t = time.time()
        with sv.managed_session(server.target) as sess:
            ims,lbls = load_tr_set(task_index, n_tasks)
            step = 0
            num_ex = lbls.shape[0]
            while not sv.should_stop() and step < 100:                
                start = (step * batch_size) % num_ex
                stop = (start + batch_size) % num_ex
                slice_ = slice(start, stop)
                batch = (ims[slice_], lbls[slice_])
                if step%10 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1]})
                    print "task %d step %d, training accuracy %g"%(task_index, step, train_accuracy)
                _, step = sess.run([train_op, global_step],feed_dict={x: batch[0], y_: batch[1]})
     
        time_end = time.time()
        print "time taken: ", time_end - t
        sv.stop()

 


# In[8]:

if __name__ == "__main__":
    tf.app.run()


# In[9]:

#! ipython nbconvert --to script ./slurm_trainer.ipynb


# In[ ]:



