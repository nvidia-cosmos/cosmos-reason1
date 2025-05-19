# Directory Structure

This document outlines the purpose of the main directories and key files within the `ref-rl` project.

-   **`assets/`**: Contains static assets, potentially images or other media used by the project (e.g., for documentation or visualization).
-   **`configs/`**: Holds preseting configuration files of GRPO/SFT purpose.
-   **`cosmos_reason1/`**
    -   **`cmake/`**: Contains CMake modules or helper scripts used by `CMakeLists.txt` for building C++/CUDA components if any.
    - **`csrc/`**: Contains the C++/CUDA source code for the library.
    - **`comm/`**: Communication Mixin for replicas.
    - **`dispatcher/`**: Controller core.
    - **`patch/`**: Patch for 3rd_party libs (PyTorch, vLLM, etc).
    - **`policy/`**: Training core (Modeling, SFT/GRPO trainer, Checkpoint, Parallelism).
    - **`rollout/`**: Rollout core(Weight syncing, generation).
    - **`utils/`**: Utilization functionality.
-   **`tools/`**: Contains scripts for replica launch and evaluation.
-   **`tests/`**: Contains unit tests, integration tests, and potentially other test-related code and resources.

**Key Files:**

-   **`CMakeLists.txt`**: Build configuration file for CMake, used if the project includes C++ or CUDA code that needs compilation.
-   **`Dockerfile`**: Instructions for building a Docker container image for the project, encapsulating its environment and dependencies.
-   **`requirements.txt`**: Lists the Python package dependencies required to run the project.
-   **`ruff.toml`**: Configuration file for the Ruff linter/formatter.
-   **`setup.py`**: Script used for building, packaging, and installing the Python project (using `setuptools`).
