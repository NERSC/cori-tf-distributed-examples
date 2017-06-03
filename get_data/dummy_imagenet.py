
# coding: utf-8

# In[4]:

import numpy as np


image_shape = (224, 224, 3)
num_classes = 1000
label_shape = ()


# In[8]:

def load_tr_set(task_index, n_tasks, total_images=4000):
    images_per_task = int(total_images / n_tasks)
    images = np.random.random(size=(images_per_task, image_shape[0], image_shape[1], image_shape[2]))
    labels = np.random.randint(low=0, high=num_classes, size=(images_per_task,))
    return images, labels, total_images
    
    
    
    


# In[ ]:

def get_shapes():
    return image_shape, label_shape
    


# In[9]:

if __name__ == "__main__":
    arr =load_tr_set(3,160)


# In[5]:

#! jupyter nbconvert --to script dummy_imagenet.ipynb


# In[ ]:



