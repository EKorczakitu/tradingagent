#!/bin/bash
#SBATCH --job-name=Trade5_Ensemble_Transformer
#SBATCH --output=logs/slurm_trade_%j.txt    # Gem logs i en undermappe (hold det ryddeligt)
#SBATCH --error=logs/slurm_trade_%j.err     # Adskil fejl fra standard output
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=32                  # Opgraderet til 32 for at understøtte (5 workers * 6 threads) + system overhead
#SBATCH --mem=64G                           # Øget lidt til at håndtere 5x Transformer gradients + Replay Buffers i RAM
#SBATCH --time=40:00:00

# 1. Load Modules
module purge
module load CUDA/12.1.1
module load Python/3.12.3-GCCcore-13.3.0

# 2. Aktiver Virtual Environment (TILPAS DENNE LINJE TIL DIT SETUP)
# source /path/til/din/venv/bin/activate 

# 3. Setup TMPDIR & Sikker Cleanup (HPC Standard)
export TMPDIR=/tmp/${USER}_job_${SLURM_JOB_ID}
mkdir -p $TMPDIR
export JOBLIB_TEMP_FOLDER=$TMPDIR

# Sikkerheds-trap: Sletter TMPDIR uanset om scriptet fejler, bliver dræbt af SLURM eller fuldfører succesfuldt.
trap "echo 'Rydder op i $TMPDIR...'; rm -rf $TMPDIR; echo 'Cleanup komplet.'" EXIT

cd $SLURM_SUBMIT_DIR
mkdir -p logs # Sikr at log-mappen eksisterer

# 4. Optimer PyTorch GPU Allocation (Undgår VRAM fragmentering når 5 workers deler 1 A30)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. Kør koden
echo "=================================================="
echo "🚀 Starter Trading Agent Pipeline"
echo "Arkitektur: Time-Series Transformer + DSR Reward"
echo "Ensemble Size: 5 Modeller (ProcessPoolExecutor)"
echo "Node: $(hostname) | CPU-kerner: $SLURM_CPUS_PER_TASK | TMPDIR: $TMPDIR"
echo "=================================================="

# -u sikrer unbuffered output, så du kan følge med i .txt filen live
python3 -u main.py