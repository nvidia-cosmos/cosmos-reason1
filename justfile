default:
    just --list

install:
    uv sync --no-default-groups
    uv sync
