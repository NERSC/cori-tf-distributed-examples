# cori-tf-distributed-examples
Scripts/Benchmarks for Running Tensorflow Distributed on Cori
### Running Code
#### Running In Batch
sbatch -N \<number of nodes\> -t \<time to run\> train.sl \<command line arguments to main.py\>

#### Running Interactively
salloc -N \<number of nodes\> -t \<time to run\> -C \<haswell or knl\>

bash train.sl \<command line arguments to main.py\>

#### Command line arguments to main.py
  
usage: main.py [-h] [-m {async,sync}] [-b BATCHSIZE] [-p PATH_TO_H5]

optional arguments:
  * -h, --help            show this help message and exit
  
  * -m {async,sync}, --mode {async,sync}
                        which mode of distributed training to use: "sync" or
                        "async" (default: sync)
                        
  * -b BATCHSIZE, --batchsize BATCHSIZE
                        what batch size to use. That is, after each node gets
                        a chunk of the data, how much data each node should
                        process per iteration (default: 128)
                        
 * -p PATH_TO_H5, --path_to_h5 PATH_TO_H5
                        path to hdf5 file for training (default: $SCRATCH/cifar10/cifar_10_caffe_hdf5/train.h5)
