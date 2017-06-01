
# coding: utf-8

# In[3]:

import numpy as np

total_images = 4000
image_shape = (224, 224, 3)
num_classes = 1000


# In[8]:

def load_tr_set(task_index, n_tasks):
    images_per_task = int(total_images / n_tasks)
    images = np.random.random(size=(images_per_task, image_shape[0], image_shape[1], image_shape[2]))
    labels = np.random.randint(low=0, high=num_classes, size=(images_per_task,))
    return images, labels
    
    
    
    


# In[5]:

def get_shapes():
    lbl_shape = ()
    return image_shape, lbl_shape


# In[9]:

if __name__ == "__main__":
    arr =load_tr_set(3,160)


# In[1]:

#! jupyter nbconvert --to script dummy_imagenet.ipynb


# In[ ]:



