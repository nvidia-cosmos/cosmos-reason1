default:
    just --list

# Install in an existing environment
install:
    uv sync --only-group build --inexact
    uv sync --all-extras --inexact

# Create a new conda environment
_conda-env conda='conda':
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf .venv
    INFO=$({{ conda }} env create -y -f environment.yml --json)
    VENV=$(echo $INFO | jq -r '."prefix"')
    ln -sf $VENV .venv

# Install in a new conda environment
install-conda conda='conda':
    just -f {{ justfile() }} _conda-env {{ conda }}
    just -f {{ justfile() }} install
