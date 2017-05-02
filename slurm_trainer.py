
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
    print "hello"
    batch_size = 128
    cluster_type = sys.argv[1]
    print "hey"
    cluster, server, task_index, n_tasks, job_name = setup_cluster(cluster_type)
    print "heilo from", task_index
    if job_name == "ps":
        print "time started: ", time.time()
        server.join()
    elif job_name == "worker":
        print "hey from", task_index
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)):
            print "worker device", task_index
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
            print "managed_sess", task_index
            ims,lbls = load_tr_set(task_index, n_tasks)
            print "the shapez", ims.shape, lbls.shape, task_index
            step = 0
	    start = 0
            num_ex = lbls.shape[0]
            while step < 1000000:                
                stop = start + batch_size
		if stop > num_ex:
		    stop = num_ex
                slice_ = slice(start, stop)
                batch = (ims[slice_], lbls[slice_])
                if step%1 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1]})
                    print "task %d step %d, training accuracy %g \n"%(task_index, step, train_accuracy)
		    sys.stdout.flush()
		num_steps_to_epoch = 60000 / batch_size
		if step % num_steps_to_epoch ==0:
		    epoch = step / num_steps_to_epoch 
		    print "Task %i finished Epoch %i at Step %i \n"%(task_index, epoch, step)
		    sys.stdout.flush()	
                _, step = sess.run([train_op, global_step],feed_dict={x: batch[0], y_: batch[1]})
		start = (start + batch_size) % num_ex
		time.sleep(1)
        time_end = time.time()
        print "time taken: ", time_end - t
        sv.stop()

 


# In[6]:

if __name__ == "__main__":
    tf.app.run()


# In[5]:

#! ipython nbconvert --to script ./slurm_trainer.ipynb


# In[ ]:



