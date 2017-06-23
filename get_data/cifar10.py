
# coding: utf-8

# In[5]:

import sys
import os
from util import h5_data_splitter, BatchFetcher, ImGenerator
import numpy as np


# In[6]:

def get_generator(num_tasks, task_id, batch_size=128, path_to_h5file=None):
    if not path_to_h5file:
        assert False, "Downloading dataset not enabled at this point. Please specify path_to_h5file"
    ims, lbls = h5_data_splitter(path_to_h5file, num_tasks, task_id)
    ims = np.transpose(ims,axes=(0,2,3,1))
    bf = BatchFetcher(ims, lbls)
    gen = ImGenerator(bf, batch_size=batch_size)
    return gen


# In[7]:

def get_num_classes():
    return 10


# In[8]:

if __name__ == "__main__":
    gen = get_generator(num_tasks=100,task_id=0,path_to_h5file="/global/cscratch1/sd/racah/cifar10/cifar_10_caffe_hdf5/train.h5")
    print gen.im_shape
    for ims,lbls in gen:
        print ims.shape, lbls.shape


# In[7]:

#!jupyter nbconvert --to script cifar10.ipynb


# In[ ]:



