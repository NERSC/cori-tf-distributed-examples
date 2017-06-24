
# coding: utf-8

# In[1]:

import argparse


# In[2]:

configs = {"dataset": "cifar10", "mode":"sync", "logdir": "./logs", "batch_size": 128, "path_to_h5_file": None}


# In[3]:

def get_configs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in configs.iteritems():
        if v:
            type_ = type(v)
        else:
            type_ = str
        parser.add_argument("--" + k,default=v, type=type_)
    args = parser.parse_args()
    configs.update(args.__dict__)
    return configs
    


# In[5]:

if __name__ == "__main__":
    cfg = get_configs()
    print cfg


# In[8]:

#! jupyter nbconvert --to script configs.ipynb


# In[ ]:



