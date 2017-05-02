#!/bin/bash -l
module load deeplearning

python slurm_trainer.py local ps 0 &> out_ps.txt &

for task_id in {0..1}
do
echo $task_id
python slurm_trainer.py local worker $task_id &> out_"$task_id".txt  &
done
