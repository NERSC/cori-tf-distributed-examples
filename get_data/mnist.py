


import os
import sys
#get mnist

import numpy as np
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import gzip
import sys
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import time

def _read32(bytestream):
    dt = np.dtype(numpy.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D unit8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print 'Extracting', f.name 
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
        raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D unit8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels

def load_tr_set(task_index, num_tasks, seed=3):
    #grab 1./num_tasks proportion of data
    with open("/global/u1/r/racah/projects/practice/tensorflow_practice//distributed/MNIST_data/train-images-idx3-ubyte.gz") as f:

        ims = extract_images(f)
    with open("/global/u1/r/racah/projects/practice/tensorflow_practice//distributed/MNIST_data/train-labels-idx1-ubyte.gz") as f:
        lbls = extract_labels(f)
    num_ex = lbls.shape[0]
    num_ex_per_task = int(num_ex / float(num_tasks))
    
    start = (task_index) * num_ex_per_task
    stop = start + num_ex_per_task
    slice_ = slice(start,stop)
    ims = shuffle_data(seed,ims)
    lbls = shuffle_data(seed, lbls)
    lbls = dense_to_one_hot(lbls, num_classes=10)
    print ims.shape
    print lbls.shape
    
    
    return ims[slice_], lbls[slice_]

def get_placeholders():
    x = tf.placeholder(tf.float32, shape=[None, 28, 28,1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    return x,y_



def shuffle_data(seed, data):
    rng =np.random.RandomState(seed)
    inds = np.arange(data.shape[0])
    rng.shuffle(inds)
    data = data[inds]
    return data

