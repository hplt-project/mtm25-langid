#!/bin/bash
#SBATCH --job-name=download_flores
#SBATCH --output=logs/download_flores_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

source env/bin/activate
python3 scripts/download_flores_plus.py
