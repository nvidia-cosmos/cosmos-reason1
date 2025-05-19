#!/bin/bash

#!/usr/bin/env bash

# Default values
NGPU=2
LOG_RANKS=""
RDZV_ENDPOINT="localhost:0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

print_help() {
  echo ""
  echo "Usage: ./launch_rollout_bench.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --ngpus <int>                      Number of GPUs per node. Default: 2"
  echo "  --config <path>                    Path to the cosmos config. Default: None"
  echo "  --number <int>                     Number of iterations. Default: -1(run until the end of the dataset)"
  echo "  --log-rank <comma-separated ints>  Comma-separated list of ranks to enable logging. Default: Empty for all ranks."
  echo "  --rdzv-endpoint <host:port>        Rendezvous endpoint for distributed training. Default: localhost:0"
  echo "  --help                             Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./launch_rollout_bench.sh --ngpus 4 --log-rank 0,1 --config <path> --number <int>"
  echo ""
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --ngpus)
    NGPU="$2"
    shift 2
    ;;
  --log-rank)
    LOG_RANKS="$2"
    shift 2
    ;;
  --config)
    CONFIG="$2"
    shift 2
    ;;
  --number)
    ITERATIONS="$2"
    shift 2
    ;;
  --rdzv-endpoint)
    RDZV_ENDPOINT="$2"
    shift 2
    ;;
  --help)
    print_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    print_help
    exit 1
    ;;
  esac
done

if [ -z "$CONFIG" ]; then
  echo "Error: --config is required"
  print_help
  exit 1
fi

if [ -z "$ITERATIONS" ]; then
  echo "Warning: --number is not set, using default value -1"
  ITERATIONS=-1
fi

torchrun --nproc-per-node="$NGPU" \
  --local-ranks-filter "$LOG_RANKS" \
  --role rank --tee 3 --rdzv_backend c10d --rdzv_endpoint="$RDZV_ENDPOINT" \
  $SCRIPT_DIR/rollout_benchmark.py --config "$CONFIG" --number "$ITERATIONS"