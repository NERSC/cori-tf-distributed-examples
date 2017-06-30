import tensorflow as tf
import sys
from models.resnet_cifar import make_model
from configs import get_configs
from slurm_tf_helper.setup_clusters import setup_slurm_cluster
import os
import numpy as np
import time
import importlib
from get_data.util import h5_data_splitter, BatchFetcher, ImGenerator
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--mode", default="sync", type=str,
                    help="which mode of distributed training to use: \"sync\" or \"async\"", 
                    choices=["async", "sync"])

parser.add_argument("-b","--batchsize", default=128, type=int,
                    help="what batch size to use. That is, after each node gets a chunk of the data, how much data each node should process per iteration")
parser.add_argument("-p", "--path_to_h5",help="path to hdf5 file for training",type=str,
                    default="/global/cscratch1/sd/racah/cifar10/cifar_10_caffe_hdf5/train.h5")
args = parser.parse_args()


def get_generator(num_tasks, task_id, batch_size=128, path_to_h5=None):
    if not path_to_h5:
        assert False, "Downloading dataset not enabled at this point. Please specify path_to_h5file"
    ims, lbls = h5_data_splitter(path_to_h5, num_tasks, task_id)
    ims = np.transpose(ims,axes=(0,2,3,1))
    bf = BatchFetcher(ims, lbls)
    gen = ImGenerator(bf, batch_size=batch_size)
    return gen


def main(_):
    
    num_classes = 10 #cuz cifar10
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)
    
    if job_name == "ps":
        server.join()
    
    elif job_name == "worker":
        is_chief=(task_index == 0)
        
        #whatever you use to iterate over your data
        generator = get_generator(num_tasks=num_tasks, 
                                  task_id=task_index, 
                                  batch_size=args.batchsize, 
                                  path_to_h5=args.path_to_h5)
        
        
        
        # Replica Device Setter Assigns ops to the local worker by default and stores all variables in the parameter server (ps).
        device = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,cluster=cluster)
        with tf.device(device):
            
            #set up model variables/ops (insert any tf or keras code here or whatever you use to define your model
            x,y, loss = make_model(generator.im_shape[1:], args.batchsize, num_classes)
            
            #global step that either gets updated after any node processes a batch (async) or when all nodes process a batch for a given iteration (sync)
            global_step = tf.contrib.framework.get_or_create_global_step()
            
            opt = tf.train.AdamOptimizer(0.1)
            
            if args.mode == "sync":
                #if syncm we make a data structure that will aggregate the gradients form all tasks (one task per node in thsi case)
                opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_tasks,
                               total_num_replicas=num_tasks)
            train_op = opt.minimize(loss, global_step=global_step)

        

        
        
        #a hook that will stop training at
        hooks=[tf.train.StopAtStepHook(last_step=1000000)]
        
        if args.mode == "sync":
            hooks.append(opt.make_session_run_hook(is_chief=is_chief))
            
        
        with tf.train.MonitoredTrainingSession(is_chief=is_chief,
                                               master=server.target,
                                               hooks=hooks) as mon_sess:
            
            
            while not mon_sess.should_stop():
                
                for ims,lbls in generator:
                    iteration_start = time.time()
                         
                    _, step, loss_ = mon_sess.run([train_op, global_step, loss],
                                                  feed_dict={x:ims, y: lbls})
                    print "Global Step is %i" % step
                    print "Loss for task id %i is: %8.4f " % (task_index, loss_)
                    print "Remember the iteration losses won't go down right away! Wait to see if the epoch losses go down, you eager beaver!"
                    print "Time for iteration for task %i, batch size %i is %5.2f" % (task_index, args.batchsize, time.time() - iteration_start)
                


if __name__ == "__main__":
    tf.app.run(main=main)



