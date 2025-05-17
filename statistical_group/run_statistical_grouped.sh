# statistical grouped model runner
#
# this script is to provide a command-line interface for running 
# statistical_group.py, handling parameter configurations and passing them to
# the Python script
#
# how to use:
# ./run_statistical_grouped.sh [options]
#
# ex.
# ./run_statistical_grouped.sh --fast # quick training with only 50 iterations
# ./run_statistical_grouped.sh --thorough # more thorough training with 200 iterations
# ./run_statistical_grouped.sh --more-groups # this groups codes more aggressively
#       (min_count=10)
# ./run_statistical_grouped.sh --max-iter 150 # custom number of iterations
# ./run_statistical_grouped.sh --min-count 8 # custom grouping threshold

# begin by setting default options
MODEL_FILE="tfidf_logreg_classifier_grouped.pkl" # would be our output model filename
MAX_ITER=100 # default number of training iterations
N_JOBS=4 # default number of parallel cores to use
REPROCESS=false # whether to reprocess the raw data
MIN_COUNT=5 # minimum frequency threshold for code grouping
USE_CUSTOM_WEIGHTS=false # whether to use custom class weights

# function to print help
function print_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help                 Display this help message"
    echo "  --model <file>         Model file to use (default: $MODEL_FILE)"
    echo "  --max-iter <num>       Maximum iterations for training (default: $MAX_ITER)"
    echo "  --n-jobs <num>         Number of parallel jobs (-1 for all cores, default: $N_JOBS)"
    echo "  --reprocess            Force reprocessing of raw data"
    echo "  --min-count <num>      Minimum count to keep a code as-is (default: $MIN_COUNT)"
    echo "  --custom-weights       Use custom class weights instead of 'balanced'"
    echo "  --fast                 Use faster settings (fewer iterations)"
    echo "  --thorough             Use more thorough settings (more iterations)"
    echo "  --more-groups          Group more aggressively (higher min-count)"
    echo "  --less-groups          Group less aggressively (lower min-count)"
    echo ""
}

# parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)
            print_help
            exit 0
            ;;
        --model)
            MODEL_FILE="$2"  # set custom model filename
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"    # set custom iteration count
            shift 2
            ;;
        --n-jobs)
            N_JOBS="$2"      # set number of parallel processes
            shift 2
            ;;
        --reprocess)
            REPROCESS=true   # force raw data reprocessing
            shift
            ;;
        --min-count)
            MIN_COUNT="$2"   # set custom grouping threshold
            shift 2
            ;;
        --custom-weights)
            USE_CUSTOM_WEIGHTS=true  # use frequency-based class weights
            shift
            ;;
        --fast)
            # preset for quick training (fewer iterations than default)
            MAX_ITER=50
            shift
            ;;
        --thorough)
            # preset for more thorough training (more iterations than default)
            MAX_ITER=200
            shift
            ;;
        --more-groups)
            # preset for more aggressive grouping (fewer classes)
            MIN_COUNT=10
            shift
            ;;
        --less-groups)
            # preset for less aggressive grouping (more classes)
            MIN_COUNT=3
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# after parsing command line arguments, we construct the command with all the params
CMD="python3 statistical_group.py --model_file $MODEL_FILE --max_iter $MAX_ITER --n_jobs $N_JOBS --min_count $MIN_COUNT"

# add optional flags if enabled
if $REPROCESS; then
    CMD="$CMD --reprocess"
fi

if $USE_CUSTOM_WEIGHTS; then
    CMD="$CMD --use_custom_weights"
fi

# print and run the command
echo "Running: $CMD"
eval $CMD

# a safety check for errors in execution
if [ $? -ne 0 ]; then
    echo "Error: Training failed. Check the output above for details."
    exit 1
else
    echo "Training completed successfully!"
fi
