#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# requires-python = ">=3.10"
# dependencies = [
# ]
# [tool.uv]
# exclude-newer = "2025-08-05T00:00:00Z"
# ///

import argparse
import glob
import json
import os
import sys  # Used for printing errors to the standard error stream

"""This script calculates the mean accuracy from a directory containing multiple JSON evaluation files.

Each JSON file is expected to contain a list of evaluation results.
Each item within the list should be an object (dictionary) that includes an 'is_correct' boolean field.
"""

if __name__ == "__main__":
    # --- Argument Parsing ---
    # Define and parse command-line arguments required by the script.
    parser = argparse.ArgumentParser(
        description="Calculate accuracy from JSON evaluation files"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory containing JSON evaluation files to be processed",
    )
    args = parser.parse_args()

    # The directory path provided by the user through the --result_dir argument.
    result_dir = args.result_dir

    # --- File Discovery ---
    # Search for all files ending with the .json extension within the specified directory.
    json_files = glob.glob(os.path.join(result_dir, "**", "*.json"), recursive=True)

    # Check if any JSON files were found. Exit if the directory is empty or contains no JSONs.
    if not json_files:
        raise RuntimeError(f"No JSON files found in directory: {result_dir}")

    # --- Data Aggregation ---
    # Initialize counters to accumulate the total number of correct samples and all processed samples
    # across all JSON files.
    total_correct = 0
    total_samples = 0

    # Process each JSON file found in the directory.
    for json_file in json_files:
        print(
            f"Processing file: {json_file}"
        )  # Provide feedback on which file is being processed

        # --- File Processing & Error Handling ---
        try:
            # Open and load data from the current JSON file. Using UTF-8 encoding is standard practice.
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

                # Validate that the loaded data is a list, as expected for iteration.
                if not isinstance(data, list):
                    print(
                        f"Warning: Skipping file '{json_file}'. Expected the top level to be a list of items, but found {type(data).__name__}.",
                        file=sys.stderr,
                    )
                    continue  # Skip this file and move to the next one

                # Iterate through each item (expected to be an evaluation result) in the loaded list.
                for item in data:
                    # Check if the item is a dictionary and contains the crucial 'is_correct' key.
                    if isinstance(item, dict) and "is_correct" in item:
                        # Increment the counter for total samples processed.
                        total_samples += 1
                        # If the 'is_correct' field is true, increment the counter for correct samples.
                        if item["is_correct"]:
                            total_correct += 1
                    # Note: Items that are not dictionaries or do not have the 'is_correct' key are ignored
                    # and do not contribute to the sample counts.

        # Handle specific errors that might occur during file operations or JSON parsing.
        except FileNotFoundError:
            print(
                f"Error: File not found while attempting to process '{json_file}'.",
                file=sys.stderr,
            )
        except json.JSONDecodeError:
            print(
                f"Error: Could not decode JSON from file '{json_file}'. Please ensure the file is valid JSON.",
                file=sys.stderr,
            )
        except Exception as e:
            # Catch any other unexpected errors during the processing of a file.
            print(
                f"An unexpected error occurred while processing '{json_file}': {e}",
                file=sys.stderr,
            )

    # --- Results Calculation & Output ---
    # Calculate the mean accuracy. Use a float division. If no samples were processed, accuracy is 0.0.
    accuracy = float(total_correct) / total_samples if total_samples > 0 else 0.0

    # Print the final aggregated results collected from all processed files.
    print(
        "\n--- Aggregated Results ---"
    )  # Add a header for clarity of the final output
    print(f"Total samples processed across all files: {total_samples}")
    print(f"Total samples counted as correct: {total_correct}")
    # Print the mean accuracy formatted to 4 decimal places and as a percentage to 2 decimal places.
    print(f"Mean accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
