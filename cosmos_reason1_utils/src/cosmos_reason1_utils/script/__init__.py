import os
import resource
import warnings

def init_script(verbose: bool = False):
    """Initialize inference script."""
    # Suppress core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Tokenizers parallelism doesn't work with vllm
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not verbose:
        warnings.filterwarnings("ignore")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
