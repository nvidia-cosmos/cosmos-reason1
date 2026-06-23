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

"""ITS Evaluation script for Cosmos-RL.

Example:

```shell
cosmos-rl-evaluate --config cosmos_rl/evaluation/configs/its_evaluate.toml
```
"""

import argparse
import toml
import logging
from pathlib import Path
from typing import Dict, Any

from cosmos_rl.utils.decorators import monitor_status
from cosmos_rl.utils.tao_status_logger import log_tao_status
from nvidia_tao_core.loggers.logging import get_status_logger, Status, Verbosity

# Constants
COMPONENT_NAME = "Cosmos-RL Evaluation"
SEPARATOR = "-" * 50

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cosmos-RL Evaluation Script")

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to evaluation configuration TOML file"
    )
    return parser.parse_args()


def run_evaluation(args):
    """
    Run the evaluation pipeline.

    Args:
        args: Parsed command line arguments
    """

    # Get status logger for TAO integration
    s_logger = get_status_logger()

    try:
        s_logger.write(
            status_level=Status.RUNNING,
            message="Starting evaluation...",
            verbosity_level=Verbosity.INFO
        )

        # Load configuration
        config_path = Path(args.config)
        logger.info(f"Loading configuration from {config_path}")

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            eval_config = toml.load(f)

        # Configure GPU usage based on num_gpus
        num_gpus = eval_config.get("num_gpus", 1)

        # Default strategy: data parallelism (tp_size=1, total_shard=num_gpus)
        # Unless tp_size is explicitly set in model config
        if "model" not in eval_config:
            eval_config["model"] = {}

        tp_size = eval_config["model"].get("tp_size", 1)

        # Calculate total_shard based on num_gpus and tp_size
        # Formula: num_gpus = total_shard × tp_size
        calculated_total_shard = num_gpus // tp_size

        # Override total_shard in evaluation config if not explicitly set
        if "evaluation" not in eval_config:
            eval_config["evaluation"] = {}

        eval_config["evaluation"]["total_shard"] = calculated_total_shard

        if "num_gpus" in eval_config and eval_config["num_gpus"] is not None:
            # Apply evaluate.num_gpus from config to model.tp_size
            num_gpus = eval_config["num_gpus"]
            if "model" not in eval_config:
                eval_config["model"] = {}
            eval_config["model"]["tp_size"] = num_gpus
            logger.info(f"Using {num_gpus} GPUs from evaluate.num_gpus config")

        # Create results directory
        results_dir = eval_config.get("results_dir", "/results")
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")

        # Initialize evaluator
        task_type = eval_config.get("task", {}).get("type", "its_directionality")
        logger.info(f"Initializing evaluator for task: {task_type}")
        lora_enabled = eval_config.get("model", {}).get("enable_lora", False)
        if task_type == "its_directionality":
            from cosmos_rl.evaluation.its_evaluator import ITSEvaluator
            evaluator = ITSEvaluator(eval_config, enable_lora=lora_enabled)
        elif task_type == "metropolis_sgd":
            from cosmos_rl.evaluation.metropolis_sgd_evaluator import MetropolisSGDEvaluator
            evaluator = MetropolisSGDEvaluator(eval_config, enable_lora=lora_enabled)
        else:
            from cosmos_rl.evaluation.evaluator import Evaluator
            evaluator = Evaluator(eval_config, enable_lora=lora_enabled)

        # Log evaluation start to TAO
        log_tao_status(
            data={
                "evaluation_status": "started",
                "config": str(config_path),
                "results_dir": str(results_dir),
                "model_name": eval_config.get("model", {}).get("model_name", "unknown")
            },
            component_name=COMPONENT_NAME
        )

        logger.info(SEPARATOR)
        logger.info("STARTING EVALUATION PIPELINE")
        logger.info(SEPARATOR)
        logger.info(f"Task: {task_type}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Results: {results_dir}")
        logger.info(f"Model: {eval_config.get('model', {}).get('model_name', 'unknown')}")
        logger.info(SEPARATOR)

        # Get evaluation parameters from config
        eval_params = eval_config.get("evaluation", {})
        skip_saved = eval_params.get("skip_saved", False)
        limit = eval_params.get("limit", -1)
        total_shard = eval_params.get("total_shard", 1)
        shard_id = eval_params.get("shard_id", 0)

        # Run evaluation pipeline (inference + scoring)
        logger.info("Starting evaluation pipeline...")
        results = evaluator.run_evaluation(
            results_dir=results_dir,
            skip_saved=skip_saved,
            limit=limit,
            total_shard=total_shard,
            shard_id=shard_id
        )

        # Extract metrics
        overall_metrics = results.get("overall", {})
        overall_accuracy = overall_metrics.get("accuracy", 0.0)
        total_samples = overall_metrics.get("total", 0)
        correct_samples = overall_metrics.get("correct", 0)

        # Prepare KPI data with all metrics flattened for visibility
        kpi_data = {
            "evaluation_status": "completed",
            "accuracy": overall_accuracy,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "results_path": str(results_dir),
        }

        # Add soft accuracy metrics if present
        if "soft_accuracy" in overall_metrics:
            kpi_data["soft_accuracy"] = overall_metrics["soft_accuracy"]
        if "soft_correct" in overall_metrics:
            kpi_data["soft_correct"] = overall_metrics["soft_correct"]
        if "soft_threshold" in overall_metrics:
            kpi_data["soft_threshold"] = overall_metrics["soft_threshold"]

        # Add weighted score if present (Metropolis SGD)
        if "weighted_score" in overall_metrics:
            kpi_data["weighted_score"] = overall_metrics["weighted_score"]

        # Add text similarity metrics (BLEU, ROUGE, BERTScore) to top level
        for key, value in results.items():
            if key not in ["overall", "metrics"] and not isinstance(value, dict):
                # Add top-level metrics like BLEU, ROUGE1, ROUGE2, etc.
                kpi_data[key.lower()] = value

        # Add per-category metrics (for Metropolis SGD, ITS, etc.)
        for category, cat_metrics in results.items():
            if category not in ["overall", "metrics", "detailed_results"] and isinstance(cat_metrics, dict):
                # Flatten category metrics: e.g., count_accuracy, distance_accuracy
                if "accuracy" in cat_metrics:
                    kpi_data[f"{category}_accuracy"] = cat_metrics["accuracy"]
                if "total" in cat_metrics:
                    kpi_data[f"{category}_total"] = cat_metrics["total"]
                if "correct" in cat_metrics:
                    kpi_data[f"{category}_correct"] = cat_metrics["correct"]

        # Keep detailed_results for backward compatibility
        kpi_data["detailed_results"] = results

        # Log final results to TAO
        log_tao_status(
            data=kpi_data,
            component_name=COMPONENT_NAME
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results saved to: {results_dir}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        print(f"Total samples: {total_samples}")
        print(f"Correct samples: {correct_samples}")
        print()

        # Print per-category results
        print("Per-category results:")
        for category, metrics in results.items():
            if category != "overall" and isinstance(metrics, dict):
                accuracy = metrics.get("accuracy", 0.0)
                total = metrics.get("total", 0)
                correct = metrics.get("correct", 0)
                print(f"  {category:<15}: {accuracy:.4f} ({correct:>3}/{total:<3})")

        print("=" * 60)

        s_logger.write(
            status_level=Status.SUCCESS,
            message=f"Evaluation completed successfully. Accuracy: {overall_accuracy:.4f}",
            verbosity_level=Verbosity.INFO
        )

    except KeyboardInterrupt:
        s_logger.write(
            status_level=Status.FAILURE,
            message="Evaluation was interrupted by user (Ctrl+C)",
            verbosity_level=Verbosity.WARNING
        )
        log_tao_status(
            data={
                "evaluation_status": "interrupted",
                "error": "User interrupted"
            },
            component_name=COMPONENT_NAME
        )
        raise

    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        s_logger.write(
            status_level=Status.FAILURE,
            message=error_msg,
            verbosity_level=Verbosity.ERROR
        )
        log_tao_status(
            data={
                "evaluation_status": "failed",
                "error": str(e)
            },
            component_name=COMPONENT_NAME
        )
        logger.error(error_msg)
        raise


@monitor_status(name="Cosmos-RL Evaluation", mode="evaluate")
def main():
    """Main entry point for the cosmos-rl-evaluate command."""
    args = parse_args()

    # Execute the evaluation function with status monitoring
    run_evaluation(args)


if __name__ == "__main__":
    main()