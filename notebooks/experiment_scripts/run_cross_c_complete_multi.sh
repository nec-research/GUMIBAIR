CONFIG_FILE="../configs/experiment_configs/cross_c_complete_multi.yaml"
DATA_DIR=$1


time python ../run_experiments.py cross_cohort $CONFIG_FILE $DATA_DIR \
    --mode complete \
    --replicates 5 \
    --benchmark \
    --output-path ../results/cross_cohort_results/cross_c_complete_multi/ \
    --cross-validate \
    --monitor-test-auc