#!/bin/bash
# Example Slurm job: extract/evaluate one (dataset, architecture, split) cell on a GPU node.
# Training and embedding extraction for the paper ran on a Slurm GPU cluster; this is a
# generic template -- replace every <PLACEHOLDER> for your site.
#SBATCH --job-name=mood-cell
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

source <PATH_TO_CONDA>/etc/profile.d/conda.sh
conda activate <ENV_NAME>
cd <PATH_TO_REPO>          # paths themselves are configured in configs/paths.yaml

DATASET=matbench_phonons ARCH=alignn SPLIT=comp_oxide

python scripts/run_detectors.py       --dataset "$DATASET" --architecture "$ARCH" --split "$SPLIT"
python scripts/run_extra_detectors.py --dataset "$DATASET" --architecture "$ARCH" --split "$SPLIT"
python scripts/run_linear_probe.py    --dataset "$DATASET" --architecture "$ARCH" --split "$SPLIT"
python scripts/run_mlp_probe.py       --dataset "$DATASET" --architecture "$ARCH" --split "$SPLIT"
