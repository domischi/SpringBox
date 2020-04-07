#!/bin/bash
#SBATCH --job-name="Springbox"
#SBATCH --ntasks=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-mail-address>
#SBATCH --output=out
#SBATCH --error=out
#======START===============================
echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "A total of $SLURM_NTASKS tasks is used"
CMD="python3 mpi-sweep.py"
MPICMD="mpirun -n $SLURM_NTASKS $CMD"
echo $CMD
echo $MPICMD
$MPICMD
#======END================================= 
