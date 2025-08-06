#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#

# Lock all Python scripts using uv.

set -euo pipefail

# Find all files with the specified shebang
files=$(grep -l '#!/usr/bin/env -S uv run --script' "$@")

# Lock files that are out of date and track if any changes were made
changes_made=false
for file in $files; do
  if ! uv lock --check --script "$file" &> /dev/null; then
    echo "Updating lock file for $file..." >&2
    uv lock --script "$file"
    changes_made=true
  fi
done

# If any lock files were updated, exit with an error code
if [ "$changes_made" = true ]; then
  exit 1
fi
