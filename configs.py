
# coding: utf-8

# In[11]:

import argparse


# In[12]:

configs = {"dataset": "dummy_imagenet", "parallelism": "async", "logdir": "./logs", "exp_name": "experiment"}


# In[8]:

def get_configs():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in configs.iteritems():
        parser.add_argument("--" + k,default=v, type=type(v))
    args = parser.parse_args()
    configs.update(args.__dict__)
    return configs
    


# In[ ]:

if __name__ == "__main__":
    cfg = get_configs()
    print cfg


# In[19]:

#! jupyter nbconvert --to script configs.ipynb


# In[ ]:



