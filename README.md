# cori-tf-distributed-examples
Scripts/Benchmarks for Running Tensorflow Distributed on Cori
### Running 
sbatch -N \<number of nodes\> -t \<time to run\> train.sl \<command line arguments to main.py\>

#### Command line arguments to main.py
  --exp_name EXP_NAME (name of experiment you are running)
  
  --logdir LOGDIR (directory to store logs for Tensorboard)
  
  --parallelism PARALLELISM (async or sync (only async is implemented))
  
  --dataset DATASET (mnist or dummy_imagenet)
  
* running without any arguments is good for starters

### Visualizing Plots

#### On Cori:
module load deeplearning

tensorboard --logdir ./logs --port 6006

#### On your local
* In a terminal:
    * ssh -L 8181:localhost:6006 cori

* In your browser:
   * localhost:8181
   
The "iter_times" plot will show the average iteration time on the y axis and the number of nodes on the x axis



