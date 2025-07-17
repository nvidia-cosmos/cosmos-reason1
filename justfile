default:
    just --list

# Install in an existing environment
install:
    uv sync --only-group build --inexact
    uv sync --all-extras --inexact

# Create a new conda environment
_conda-env:
    rm -rf .venv
    conda env create -y --no-default-packages -f conda.yaml
    ln -sf "$(conda info --base)/envs/cosmos-reason1" .venv

# Install in a new conda environment
install-conda:
    just -f {{ justfile() }} _conda-env
    just -f {{ justfile() }} install
