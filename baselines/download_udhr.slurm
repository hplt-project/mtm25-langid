#!/bin/bash
#SBATCH --job-name=download_udhr
#SBATCH --output=logs/download_udhr_%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

source env/bin/activate
python3 scripts/download_udhr.py
