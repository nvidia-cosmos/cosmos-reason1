default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Run linting and formatting
lint: setup
  pre-commit run --all-files || pre-commit run --all-files

# Run tests
test: lint
  cd cosmos_reason1_utils && uv run pytest
