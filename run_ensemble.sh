#!/bin/bash
#SBATCH --job-name=Trade7_Ensemble
#SBATCH --output=trade_output_%j.txt
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=40:00:00

# 1. Load Modules
module purge
module load CUDA/12.1.1
module load Python/3.12.3-GCCcore-13.3.0

# 2. Setup TMPDIR (Vigtigt!)
export TMPDIR=/tmp/${USER}_job_${SLURM_JOB_ID}
mkdir -p $TMPDIR

export JOBLIB_TEMP_FOLDER=$TMPDIR
# 3. Gå til mappen hvor scriptet ligger
# $SLURM_SUBMIT_DIR er mappen hvor du står, når du skriver 'sbatch'
cd $SLURM_SUBMIT_DIR

# 4. Kør koden
echo "Starter Trading Agent (7 Modeller, 50 Trials)..."
echo "Kører på lokal disk: $TMPDIR"
echo "Kører på node: $(hostname)"

# Kør main.py - den importerer selv de andre filer
python3 -u main.py

rm -rf $TMPDIR
