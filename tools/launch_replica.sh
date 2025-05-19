#!/usr/bin/env bash

# Default values
NGPU=2
LOG_RANKS=""
TYPE=""
RDZV_ENDPOINT="localhost:0"

print_help() {
  echo ""
  echo "Usage: ./launch_replica.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --type <rollout|policy>            Required. Type of replica to launch."
  echo "  --ngpus <int>                      Number of GPUs per node. Default: 2"
  echo "  --log-rank <comma-separated ints>  Comma-separated list of ranks to enable logging. Default: Empty for all ranks."
  echo "  --rdzv-endpoint <host:port>        Rendezvous endpoint for distributed training. Default: localhost:0"
  echo "  --help                             Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./launch_replica.sh --type rollout --ngpus 4 --log-rank 0,1"
  echo "  ./launch_replica.sh --type policy --ngpus 8 --log-rank 0"
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
  --type)
    TYPE="$2"
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

if [ -z "$TYPE" ]; then
  echo "Error: --type is required"
  print_help
  exit 1
fi

if [ "$TYPE" == "rollout" ]; then
  MODULE="cosmos_reason1.rollout.rollout_entrance"
elif [ "$TYPE" == "policy" ]; then
  MODULE="cosmos_reason1.policy.train"
else
  echo "Error: Invalid --type value '$TYPE'. Must be 'rollout' or 'policy'."
  print_help
  exit 1
fi

if [ -z "$COSMOS_CONTROLLER_HOST" ]; then
  echo "Error: COSMOS_CONTROLLER_HOST is not set. Please pass it in like:"
  echo "  COSMOS_CONTROLLER_HOST=<controller_host>:<controller_port> ./launch_replica.sh"
  exit 1
fi

TORCHRUN_CMD=(
  torchrun
  --nproc-per-node="$NGPU"
  --role rank
  --tee 3
  --rdzv_backend c10d
  --rdzv_endpoint="$RDZV_ENDPOINT"
)

if [ -n "$LOG_RANKS" ]; then
  TORCHRUN_CMD+=(--local-ranks-filter "$LOG_RANKS")
fi

TORCHRUN_CMD+=(
  -m "$MODULE"
)

"${TORCHRUN_CMD[@]}"
