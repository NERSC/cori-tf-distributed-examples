


import numpy as np

total_images = 1.2 * 10**6
image_shape = (224, 224, 3)
num_classes = 1000






def load_tr_set(task_index, n_tasks):
    images_per_task = int(total_images / n_tasks)
    images = np.random.random(size=(images_per_task, image_shape[0], image_shape[1], image_shape[2]))
    labels = np.random.randint(low=0, high=num_classes, size=(images_per_task,1))
    return images
    
    
    
    



if __name__ == "__main__":
    arr =load_tr_set(3,160)





