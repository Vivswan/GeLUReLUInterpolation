#!/bin/bash
#SBATCH --job-name=run_@@@RunScript@@@_@@@run_combination@@@
#SBATCH --output=slurm_%x_%A_%a.log
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --chdir="/ihome/nyoungblood/vis77/logs"
#SBATCH --array=@@@array@@@
#SBATCH --exclude=gpu-n[31,32]
#SBATCH --requeue

source ~/.bashrc

RunDirectoryLocation="${HOME}/storage/GeLUReLUInterpolation"
StorageDirectory="${HOME}/storage/"

cd ~ || exit
conda activate geluInterpolation

SlurmScratchName="slurm_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
SlurmScratchDirectory="${SLURM_SCRATCH}/${SlurmScratchName}"
rsync -ar "${RunDirectoryLocation}"/ "${SlurmScratchDirectory}"/
rm -rf "${SlurmScratchDirectory}"/_results
rm -rf "${SlurmScratchDirectory}"/_data
mkdir -p "${SlurmScratchDirectory}"
mkdir -p "${SlurmScratchDirectory}"/_results
mkdir -p "${SlurmScratchDirectory}"/_results/datasets/
mkdir -p "${SlurmScratchDirectory}"/_results/runtime/
mkdir -p "${SlurmScratchDirectory}"/_results/models/
mkdir -p "${SlurmScratchDirectory}"/_results/tensorboard/
mkdir -p "${SlurmScratchDirectory}"/_results/logs/
rsync -ar "${HOME}"/storage/_datasets/ "${SlurmScratchDirectory}"/_results/datasets/

run_on_exit(){
  echo "####################################### Exit Began #######################################"
  rm -rf "${SlurmScratchDirectory}/_results/datasets/"
  cd "${SlurmScratchDirectory}" || exit
  srun tar -czf "${SlurmScratchName}.tar.gz" _results
  srun mv       "${SlurmScratchName}.tar.gz" "${StorageDirectory}"

  echo ""
  echo "####################################### Billing #######################################"
  echo ""
  sacct -M "$SLURM_CLUSTER_NAME" -j $SLURM_JOBID --format=AllocTRES%50,elapsed
  echo ""

  echo ""
  echo "####################################### crc-job-stats.py #######################################"
  echo ""
  crc-job-stats
  echo ""
  echo "!!!!!!Completed!!!!!!!"
  echo ""
}
trap run_on_exit EXIT

cd "${SlurmScratchDirectory}" || exit

echo ""
echo "####################################### Variables #######################################"
echo ""
echo "SlurmScratchName = ${SlurmScratchName}"
echo "StorageDirectory = ${StorageDirectory}"
echo "SlurmScratchDirectory = ${SlurmScratchDirectory}"
echo "RunDirectoryLocation = ${RunDirectoryLocation}"
echo "Array ID = ${SLURM_ARRAY_TASK_ID}"
echo ""
echo ""

echo ""
echo "####################################### Main Program: Starting #######################################"
echo ""

ArrayTaskIdOffset=@@@ArrayTaskIdOffset@@@
TotalArrayTaskId=$((SLURM_ARRAY_TASK_ID + ArrayTaskIdOffset))

srun python3 @@@RunScript@@@.py \
--data_folder ./_results \
--run_combination @@@run_combination@@@ \
--run_index ${TotalArrayTaskId} \
--tensorboard \
--save_data \


echo ""
echo "####################################### Main Program: Finished #######################################"
echo ""

