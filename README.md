# cori-tf-distributed-examples
Scripts/Benchmarks for Running Tensorflow Distributed on Cori

### Running 

#### Download Cifar10
python get_data/download-and-convert-cifar-10.py

#### Submit Job
sbatch -N \<number of nodes\> -t \<time to run\> train.sl \<command line arguments to main.py\>

#### Command line arguments to main.py
  
  --logdir LOGDIR (directory to store logs for Tensorboard)
  
  --mode ("async" or "sync")
  
  --dataset ("cifar10")
  
  --path_to_h5_file (path to where the hdf5 file from download-and-convert-cifar-10.py is)
  
  --batch_size
