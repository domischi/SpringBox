#!/bin/bash
#SBATCH --job-name="Springbox"
#SBATCH --ntasks=8
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/out
#SBATCH --error=/tmp/out
#======START===============================
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"
echo "Using $SLURM_CPUS_PER_TASK cpus per task"
CMD="python3 mpi-sweep.py"
MPICMD="mpirun -n $SLURM_NTASKS $CMD"
echo $CMD
echo $MPICMD
$MPICMD
#======END================================= 
