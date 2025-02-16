CONFIG_FILE="../configs/experiment_configs/in_c.yaml"
DATA_DIR=$1

time python ../run_experiments.py in_cohort $CONFIG_FILE $DATA_DIR \
    --replicates 5 \
    --benchmark \
    --output-path ./ \
    --cross-validate