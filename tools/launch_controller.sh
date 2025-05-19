#!/usr/bin/env bash

PORT=""
CONFIG_FILE=""
LOG_FILE=""

show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  --port <port>       Specify the port number (if not set, automatically chosen in runtime)"
  echo "  --config <file>     Specify the configuration file"
  echo "  --log <file>        Specify the redis log file"
  echo "  --help              Show this help message and exit"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

CMD="python -m cosmos_reason1.dispatcher.run_web_panel"

if [[ -n "$PORT" ]]; then
  CMD+=" --port $PORT"
fi

if [[ -n "$CONFIG_FILE" ]]; then
  CMD+=" --config $CONFIG_FILE"
fi

if [[ -n "$LOG_FILE" ]]; then
  CMD+=" --redis-logfile-path $LOG_FILE"
fi

CMD+=" --create-redis-config"

echo "${CMD}"

$CMD
