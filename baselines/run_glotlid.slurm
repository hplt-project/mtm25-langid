#!/bin/bash
#SBATCH --job-name=glotlid
#SBATCH --output=logs/glotlid_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=cpu-troja,cpu-ms
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

source env/bin/activate
time scripts/run_fasttext_based.sh glotlid
