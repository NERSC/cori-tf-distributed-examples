
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

import sys

import os


# In[5]:

"""Works for tensorflow 0.12 for now"""
def setup_slurm_cluster(num_ps=1):
    all_nodes = get_all_nodes()

    port = get_allowed_port()
    
    hostlist = [ ("%s:%i" % (node, port)) for node in all_nodes ] 
    ps_hosts, worker_hosts = get_parameter_server_and_worker_hosts(hostlist, num_ps=num_ps)


    proc_id, num_procs = get_slurm_proc_variables()
    
    num_tasks = num_procs - num_ps
    
    job_name = get_job_name(proc_id, num_ps)
    
    task_index = get_task_index(proc_id, job_name, num_ps)
    
    cluster_spec = make_cluster_spec(worker_hosts, ps_hosts)

    server = make_server(cluster_spec, job_name, task_index)
    
    return cluster_spec, server, task_index, num_tasks, job_name
    

def make_server(cluster_spec, job_name, task_index):
    server = tf.train.Server(cluster_spec,
                           job_name=job_name,
                           task_index=task_index)
    return server
    
    
def make_cluster_spec(worker_hosts, ps_hosts):
    cluster_spec = tf.train.ClusterSpec({
    "worker": worker_hosts,
    "ps": ps_hosts
    })
    return cluster_spec
    
    
def get_task_index(proc_id, job_name, num_ps):
    
    if job_name == "ps":
        task_index = proc_id
    elif job_name == "worker":
        #expects a task_index for workers that starts at 0
        task_index = proc_id - num_ps
    return task_index
    
    
def get_slurm_proc_variables():
    proc_id  = int( os.environ['SLURM_PROCID'] )
    num_procs     = int( os.environ['SLURM_NPROCS'] )
    return proc_id, num_procs
    
def get_job_name(proc_id, num_ps):
    if proc_id < num_ps:
        job_name = "ps"
    else:
        job_name = "worker"
    return job_name
    
    
def get_parameter_server_and_worker_hosts(hostlist, num_ps=1):
    """assumes num_ps nodes used for parameter server (one ps per node)
    and rest of nodes used for workers"""
    ps_hosts = hostlist[:num_ps]
    worker_hosts = hostlist[num_ps:]
    return ps_hosts, worker_hosts
    
def get_allowed_port():
    allowed_port = 22222
    return allowed_port


def get_all_nodes():
    return expand_nodelist( os.environ['SLURM_NODELIST'])
    

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
    cluster, server, task_index, num_tasks, job_name = setup_slurm_cluster(num_ps=1)


# In[2]:

#! jupyter nbconvert --to script setup_clusters.ipynb


# In[ ]:



