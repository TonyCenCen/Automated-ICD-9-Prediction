# neural network model runner
#
# this script is to provide a command-line interface for running 
# nn_models.py, handling parameter configurations and passing them to
# the Python script
#
# how to use:
# ./run_nn_models.sh [options]
#
# ex.
# ./run_nn_models.sh # default CNN model
# ./run_nn_models.sh --model lstm # LSTM model
# ./run_nn_models.sh --quick # quick training with smaller model
# ./run_nn_models.sh --full # thorough training with larger model
# ./run_nn_models.sh --epochs 20 # custom number of epochs
# ./run_nn_models.sh --model cnn --min-count 8 # custom grouping threshold

# begin by setting default options
MODEL_TYPE="cnn" # default model architecture (CNN or LSTM)
MIN_COUNT=10 # minimum count for code grouping 
BATCH_SIZE=64 # number of samples per gradient update
EPOCHS=10  # number of training iterations over the dataset
SEQ_LENGTH=500  # maximum length of text sequences
EMBEDDING_DIM=100 # dimension for word embeddings
SAVE_PATH="nn_model_weights.h5"  # path to save model weights

# function to print help
function print_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --help                Display this help message"
    echo "  --model <type>        Model type (cnn or lstm, default: $MODEL_TYPE)"
    echo "  --min-count <num>     Minimum count for code grouping (default: $MIN_COUNT)"
    echo "  --batch-size <num>    Batch size for training (default: $BATCH_SIZE)"
    echo "  --epochs <num>        Number of training epochs (default: $EPOCHS)"
    echo "  --seq-length <num>    Maximum sequence length (default: $SEQ_LENGTH)"
    echo "  --embedding <num>     Embedding dimension (default: $EMBEDDING_DIM)"
    echo "  --save-path <path>    Path to save model weights (default: $SAVE_PATH)"
    echo "  --quick               Use smaller model settings for quick training"
    echo "  --full                Use larger model settings for better performance"
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
            MODEL_TYPE="$2" # set model architecture (CNN or LSTM)
            shift 2
            ;;
        --min-count)
            MIN_COUNT="$2" # set minimum frequency for code grouping
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2" # set batch size for training
            shift 2
            ;;
        --epochs)
            EPOCHS="$2" # set number of training epochs
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2" # set maximum sequence length for text
            shift 2
            ;;
        --embedding)
            EMBEDDING_DIM="$2" # set embedding dimension size
            shift 2
            ;;
        --save-path)
            SAVE_PATH="$2" # set path to save model weights
            shift 2
            ;;
        --quick)
            # a preset for quick training (smaller model, fewer epochs), which
            # is useful for testing or parameter tuning
            MODEL_TYPE="cnn" # CNN is faster than LSTM
            BATCH_SIZE=128  # larger batch size for faster training
            EPOCHS=5  # fewer epochs
            SEQ_LENGTH=300  # shorter sequence length
            EMBEDDING_DIM=50  # smaller embedding dimension
            shift
            ;;
        --full)
            # a preset for thorough training (larger model, more epochs)
            # Useful for final model training for best performance
            MODEL_TYPE="lstm" # LSTM should capture more context 
            BATCH_SIZE=32   # smaller batch size for better gradient estimates
            EPOCHS=15  # more epochs for convergence
            SEQ_LENGTH=1000  # longer sequence length to capture more text
            EMBEDDING_DIM=200  # larger embedding dimension for better representations
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
CMD="python3 nn_models.py --model_type $MODEL_TYPE --min_count $MIN_COUNT --batch_size $BATCH_SIZE --epochs $EPOCHS --seq_length $SEQ_LENGTH --embedding_dim $EMBEDDING_DIM --save_path $SAVE_PATH"

# print and run the command
echo "Running: $CMD"
eval $CMD

# a safety check for errors in execution
if [ $? -ne 0 ]; then
    # Non-zero exit code indicates an error
    echo "Error: Training failed. Check the output above for details."
    exit 1
else
    # Successful execution
    echo "Training completed successfully!"
fi
