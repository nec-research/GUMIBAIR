CONFIG_FILE="../configs/experiment_configs/cross_c_partial_all.yaml"
DATA_DIR=$1

time python ../run_experiments.py cross_cohort $CONFIG_FILE $DATA_DIR \
    --mode partial \
    --test-prop 0.9,0.7,0.6,0.5\
    --replicates 5 \
    --benchmark \
    --all-cohorts \
    --output-path ./