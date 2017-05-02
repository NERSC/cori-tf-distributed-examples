
# coding: utf-8

# In[ ]:

import tensorflow as tf


# In[ ]:

import sys
# from nbfinder import NotebookFinder
# sys.meta_path.append(NotebookFinder())
from util import expand_nodelist
import os
import random

# In[ ]:

def setup_slurm_cluster():
    tf_hostlist = [ ("%s:22222" % node) for node in
                expand_nodelist( os.environ['SLURM_NODELIST']) ] 
    ps_hosts = [tf_hostlist[0]]
    print ps_hosts
    worker_hosts = tf_hostlist[1:]
    print worker_hosts
    task_index  = int( os.environ['SLURM_PROCID'] )
    n_tasks     = int( os.environ['SLURM_NPROCS'] )
    print task_index
    
    cluster = tf.train.ClusterSpec({
    "worker": worker_hosts,
    "ps": ps_hosts
    })

    if task_index == 0:
        job_name = "ps"
    else:
        job_name = "worker"
        #expects a task_index for worker
        task_index -= 1
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)
    return cluster, server, task_index, n_tasks, job_name
    

def setup_local_cluster(num_workers):
    port_list = range(8000,9000)
    random.shuffle(port_list)
    tf_hostlist = [ (os.environ["HOSTNAME"] + ":%i" % port) for port in
                port_list]
    
    
    ps_hosts = [tf_hostlist[0]]
    worker_hosts = tf_hostlist[1:num_workers+1]
    job_name = sys.argv[2]
    task_index  = int(sys.argv[3])
    print "the task index is: ", task_index 
    n_tasks     = len(port_list)
    
    cluster = tf.train.ClusterSpec({
    "worker": worker_hosts,
    "ps": ps_hosts
    })
    
    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=job_name,
                           task_index=task_index)
    

    return cluster, server, task_index, n_tasks, job_name

def setup_cluster(typ_, num_workers=2):
    if typ_ == "slurm":
        return setup_slurm_cluster()
    else:
        return setup_local_cluster(2)

