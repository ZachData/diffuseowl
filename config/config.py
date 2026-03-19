"""
Central configuration for probes-for-diffuse-control.

All model-specific settings and paths are defined here.
To switch models, update MODEL_NAME and MODEL_SHORT_NAME.
"""

# Model configuration
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_SHORT_NAME = "gemma-3-12b"  # Used for directory/file names

# VLLM server configuration
VLLM_BASE_PORT = 8100
VLLM_MAX_MODEL_LEN = 2000
VLLM_GPU_MEMORY_UTILIZATION = 0.45  # Use ~45% for vLLM, leave rest for activation extraction
VLLM_GPUS = [0]  # Single H200 GPU - vLLM server
VLLM_NUM_SERVERS = len(VLLM_GPUS)
ACTIVATION_GPUS = [0]  # Same GPU for activation extraction (will share memory with vLLM)

# Directory structure (model-specific)
BASE_DIR = "experiments"
CACHED_ACTIVATIONS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/cached_activations"
RESULTS_DIR = f"{BASE_DIR}/{MODEL_SHORT_NAME}/results"  # Base results directory
MMLU_FILE = "mmlu_data/train.json" #"mmlu_data/test.json"

def get_results_dir(num_examples: int = None, filter_reliable: bool = False, prompt_name: str = None) -> str:
    """
    Get results directory path organized by prompt, number of examples and filter status.
    
    Args:
        num_examples: Number of examples (default: DEFAULT_NUM_EXAMPLES)
        filter_reliable: Whether using filtered data (default: False)
        prompt_name: Prompt name (e.g., "benign", "semimalign", "lie_detector"). 
                     If None or standard prompts, uses default structure.
                     Special prompts like "lie_detector" get separate directories.
        
    Returns:
        Path to results subdirectory
    """
    num_examples = num_examples or DEFAULT_NUM_EXAMPLES
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    
    # Special handling for lie_detector and potentially other specialized experiments
    if prompt_name == "lie_detector":
        return f"{RESULTS_DIR}/lie_detector/n{num_examples}_{filter_suffix}"
    else:
        # Default structure for benign, semimalign, etc.
        return f"{RESULTS_DIR}/n{num_examples}_{filter_suffix}"

# Generation parameters
USE_MULTI_TOKEN_PROMPTS = False
MAX_NEW_TOKENS = 100 #100 #900 #100
TEMPERATURE = 1.0  # Sampling temperature

# Experiment parameters
LINEAR_AUROC_VS_N_NUM_TRIALS = 1 #3 #10
MLP_AUROC_VS_N_NUM_TRIALS = 2 #3
RELIABLE_QUESTIONS_NUM_TRIALS = 3
DEFAULT_NUM_EXAMPLES = 100000
DEFAULT_LAYER = 30 #47 #30
DEFAULT_TOKEN_POSITION = "letter"
# Available experiments: ["linear_probe", "mlp_probe", "pca", "anomaly_detection", "auroc_vs_n", "corruption_sweep", "auroc_vs_n_mlp"]
DEFAULT_EXPERIMENTS = ["auroc_vs_n"] #, "linear_probe", "mlp_probe", "pca", "anomaly_detection", "corruption_sweep"] #["linear_probe", "mlp_probe", "pca", "anomaly_detection", "auroc_vs_n", "auroc_vs_n_mlp", "corruption_sweep"]
DEFAULT_PROMPT_NAME = "lie_detector"

# SUPPORTED_LAYERS = [DEFAULT_LAYER] #[0, 2, 5, 10, 15, 20, 25, 30, 36, 42, 45, 47]
SUPPORTED_LAYERS = [0, 2, 5, 10, 15, 20, 25, 30, 36, 42, 45, 47]

DEFAULT_LAYER_SWEEP = [0, 2, 5, 10, 20, 30, 36, 42, 45, 47]
# for layer in DEFAULT_LAYER_SWEEP:
#     assert layer in SUPPORTED_LAYERS, f"Layer {layer} not in supported layers"

SUPPORTED_POSITIONS = ["last", "first", "middle", "all", "all_appended", "letter", "letter+1", "letter-1", "yes_no", "yes_no-1"]
DEFAULT_POSITION_SWEEP = ["letter-1","letter", "letter+1", "yes_no-1", "yes_no", "last"]
# Positions to cache in unified cache (excludes "all" and "all_appended" which can be computed from others)
# CACHED_POSITIONS = [DEFAULT_TOKEN_POSITION]
CACHED_POSITIONS = DEFAULT_POSITION_SWEEP

# Probe training parameters
PROBE_MAX_ITER = 1000 #1000
PROBE_RANDOM_STATE = 42
PROBE_SOLVER = 'lbfgs'  # Solver for logistic regression: 'lbfgs', 'saga', 'liblinear', 'newton-cg', 'sag'
                        # Try 'liblinear' or 'sag' if lbfgs is slow on large datasets
PROBE_TOL = 1e-2        # Convergence tolerance (1e-4 is sklearn default, 1e-2 for faster training)
PROBE_N_JOBS = -1       # Number of CPU cores for parallel computation (-1 = all cores)
PROBE_VERBOSE = False   # Enable diagnostic logging to see iterations, convergence, and timing
MAX_HIGH = 100 # 2**10 = 1024

# Linear Probe Mode Selection
LINEAR_PROBE_MODE = "closed_form"  # "sklearn", "pytorch", or "closed_form"
                                    # sklearn: Logistic regression (slow for large datasets)
                                    # pytorch: PyTorch linear model with gradient descent (better scaling)
                                    # closed_form: Ridge regression closed-form solution (fastest, 10-100x speedup)

# Linear Probe parameters (used by pytorch and closed_form modes)
LINEAR_PROBE_MAX_STEPS = 10000  # Maximum training steps for PyTorch linear probe
LINEAR_PROBE_EVAL_EVERY_N_STEPS = 100  # Evaluate validation loss every N steps
LINEAR_PROBE_BATCH_SIZE = 256  # Batch size for PyTorch linear probe and closed_form predictions
LINEAR_PROBE_LEARNING_RATE = 0.01  # Learning rate for PyTorch linear probe
LINEAR_PROBE_WEIGHT_DECAY = 1e-4  # Weight decay (L2 regularization) for PyTorch
LINEAR_PROBE_PATIENCE = 10  # Early stopping patience for PyTorch

# Closed-form Ridge Regression specific parameters
RIDGE_ALPHA = 1e0 #1e-3  # Regularization strength (λ) for ridge regression (increased for numerical stability)
RIDGE_USE_DUAL = "auto"  # "auto", "primal", or "dual" form
RIDGE_DUAL_THRESHOLD = 0.5  # When use_dual="auto", use dual form only when n_samples/n_features < threshold
                            # Lower values (e.g., 0.3) = more conservative, avoids transition zone
                            # Higher values (e.g., 0.8) = more aggressive dual form usage

# Ridge Alpha Cross-Validation parameters
RIDGE_ALPHA_CV_ENABLED = True  # Enable CV for alpha selection
RIDGE_ALPHA_CV_FOLDS = 5  # Number of k-fold CV folds
RIDGE_ALPHA_CV_ALPHAS = [1e2, 1e4, 1e5, 1e6, 1e7]  # Alpha values to test

# MLP training parameters
MLP_MAX_STEPS = 10000 #10000  # Maximum number of training steps (batches) - scales better with dataset size
MLP_EVAL_EVERY_N_STEPS = 100  # Evaluate validation loss every N steps

# MLP Probe Configuration (optimal from hyperparameter search)
# Best AUROC: 0.8548 on Layer 30, Position: letter, n=2000
MLP_HIDDEN_LAYER_SIZES = (12000, 6000)  # Two-layer pyramidal architecture
MLP_LEARNING_RATE = 0.001
MLP_WEIGHT_DECAY = 0 #0.05 #0.08
MLP_DROPOUT = 0 #0.1
MLP_PATIENCE = 10
MLP_LR_SCHEDULER = None
MLP_USE_CONSTANT_RESIDUAL = True #False
MLP_BATCH_SIZE = 256  # Batch size for MLP training (reduce if OOM, increase for speed)
MLP_SAVE_CHECKPOINT_EVERY_N = 5  # Save checkpoint every Nth validation improvement

# Multiprocessing parameters
MAX_CONCURRENT_REQUESTS_PER_SERVER = 100  # For VLLM API calls
ACTIVATION_BATCH_SIZE = 1000 #100 # Batch size for activation extraction (reduce if OOM)
VLLM_CHUNK_SIZE = 1000 #100  # Size of chunks for dynamic distribution to VLLM servers

# Visualization parameters
N_ANNOTATE = 30
SAVE_PLOTS = True  # Set to False to suppress creation of .png files

# Verbosity settings
# 0 = Silent (only errors and final results)
# 1 = Minimal (key milestones only)
# 2 = Normal (default - all status updates)
# 3 = Verbose (includes debug information)
VERBOSITY = 0

def get_config():
    """Return a dictionary with all configuration settings."""
    return {
        "model_name": MODEL_NAME,
        "model_short_name": MODEL_SHORT_NAME,
        "vllm_base_port": VLLM_BASE_PORT,
        "vllm_num_servers": VLLM_NUM_SERVERS,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "vllm_gpus": VLLM_GPUS,
        "activation_gpus": ACTIVATION_GPUS,
        "cached_activations_dir": CACHED_ACTIVATIONS_DIR,
        "results_dir": RESULTS_DIR,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "default_layer": DEFAULT_LAYER,
        "default_num_examples": DEFAULT_NUM_EXAMPLES,
        "default_token_position": DEFAULT_TOKEN_POSITION,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_random_state": PROBE_RANDOM_STATE,
        "probe_solver": PROBE_SOLVER,
        "probe_tol": PROBE_TOL,
        "probe_n_jobs": PROBE_N_JOBS,
        "probe_verbose": PROBE_VERBOSE,
        "linear_probe_mode": LINEAR_PROBE_MODE,
        "linear_probe_max_steps": LINEAR_PROBE_MAX_STEPS,
        "linear_probe_eval_every_n_steps": LINEAR_PROBE_EVAL_EVERY_N_STEPS,
        "linear_probe_batch_size": LINEAR_PROBE_BATCH_SIZE,
        "linear_probe_learning_rate": LINEAR_PROBE_LEARNING_RATE,
        "linear_probe_weight_decay": LINEAR_PROBE_WEIGHT_DECAY,
        "linear_probe_patience": LINEAR_PROBE_PATIENCE,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_use_dual": RIDGE_USE_DUAL,
        "ridge_dual_threshold": RIDGE_DUAL_THRESHOLD,
        "mlp_max_steps": MLP_MAX_STEPS,
        "mlp_eval_every_n_steps": MLP_EVAL_EVERY_N_STEPS,
        "mlp_batch_size": MLP_BATCH_SIZE,
        "max_concurrent_requests_per_server": MAX_CONCURRENT_REQUESTS_PER_SERVER,
        "activation_batch_size": ACTIVATION_BATCH_SIZE,
        "vllm_chunk_size": VLLM_CHUNK_SIZE,
        "save_plots": SAVE_PLOTS,
    }


def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    config = get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()

