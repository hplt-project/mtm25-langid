#!/bin/bash
#SBATCH --job-name=download_openlid
#SBATCH --output=logs/download_openlid_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source env/bin/activate
python3 scripts/download_openlid.py
