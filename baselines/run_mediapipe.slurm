#!/bin/bash
#SBATCH --job-name=mediapipe
#SBATCH --output=logs/mediapipe_%j.out
#SBATCH --error=logs/mediapipe_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source env/bin/activate
python3 scripts/run_mediapipe.py mediapipe
