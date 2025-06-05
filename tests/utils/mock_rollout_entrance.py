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

import sys
import time
import requests
from typing import override

from cosmos_reason1.utils.logging import logger
from cosmos_reason1.utils.parallelism import ParallelDims
from cosmos_reason1.policy.config import Config as RolloutConfig
from cosmos_reason1.utils.distributed import (
    init_distributed,
    destroy_distributed,
    get_controller_metadata,
)
from cosmos_reason1.rollout.vllm_rollout.vllm_rollout_worker import vLLMRolloutWorker
from cosmos_reason1.dispatcher.protocol import RolloutRequest
from cosmos_reason1.utils.api_suffix import COSMOS_API_ROLLOUT_SUFFIX


class mock_vLLMRolloutWorker(vLLMRolloutWorker):
    """
    To compat with GRPOTrainer, we just mock the rollout procedure method.

    We intentionally generate incorrect responses so that the model can compute reward
    """

    mock_data = [
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " Let's solve the problem of finding how many natural numbers less than a thousand are not divisible by either 5 or 7.\n\nFirst, let's determine the total number of natural numbers less than 1000, which is \\(999\\).\n\nNext, we will use the principle of inclusion and exclusion to find the number of natural numbers less than 1000 that are divisible by either 5 or 7.\n\n1. **Count the numbers divisible by 5:**\n\n   The numbers divisible by 5 form an arithmetic sequence: \\(5, 10, 15, \\ldots, 995\\).\n\n   The general term of this sequence is given by:\n   \\[\n   a_n = 5n\n   \\]\n   We need \\(a_n < 1000\\):\n   \\[\n   5n < 1000 \\implies n < 200\n   \\]\n   Thus, there are \\(199\\) terms in this sequence.\n\n2. **Count the numbers divisible by 7:**\n\n   The numbers divisible by 7 form an arithmetic sequence: \\(7, 14, 21, \\ldots, 994\\).\n\n   The general term of this sequence is given by:\n   \\[\n   b_m = 7m\n   \\]\n   We need \\(b_m < 1000\\):\n   \\[\n   7m < 1000 \\implies m < \\frac{1000}{7} \\approx 142.857\n   \\]\n   Thus, there are \\(142\\) terms in this sequence.\n\n3. **Count the numbers divisible by both 5 and 7 (i.e., divisible by \\(35\\)):**\n\n   The numbers divisible by 35 form an arithmetic sequence: \\(35, 70, 105, \\ldots, 980\\).\n\n   The general term of this sequence is given by:\n   \\[\n   c_k = 35k\n   \\]\n   We need \\(c_k < 1000\\):\n   \\[\n   35k < 1000 \\implies k < \\frac{1000}{35} \\approx 28.571\n   \\]\n   Thus, there are \\(28\\) terms in this sequence.\n\n4. **Apply the principle of inclusion and exclusion:**\n\n   The number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n   \\[\n   n(\\text{divisible by 5 or 7}) = n(\\text{divisible by 5}) + n(\\text{divisible by 7}) - n(\\text{divisible by 35})\n   \\]\n   Substituting the counts we found:\n   \\[\n   n(\\text{divisible by 5 or 7}) = 199 + 142 - 28 = 313\n   \\]\n\n5. **Count the numbers less than 1000 that are not divisible by either 5 or 7:**\n\n   Subtract the number of numbers divisible by either 5 or 7 from the total number of natural numbers less than 1000:\n   \\[\n   n(\\text{not divisible by 5 or 7}) = 999 - 313 = 686\n   \\]\n\nThus, the number of natural numbers less than a thousand that are not divisible by either 5 or 7 is \\(\\boxed{684}\\).",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " To determine how many natural numbers less than a thousand are not divisible by either 5 or 7, we can use the principle of inclusion-exclusion and the principle of counting the total numbers and subtracting those that are divisible by either 5 or 7.\n\nFirst, count the total number of natural numbers less than 1000:\n\\[999.\\]\n\nNext, count the number of natural numbers less than 1000 that are divisible by 5. The sequence of such numbers is an arithmetic sequence with the first term 5 and the common difference 5. The largest number less than 1000 that is divisible by 5 is 995. The number of terms in this sequence is:\n\\[\n\\frac{995}{5} = 199.\n\\]\n\nNow, count the number of natural numbers less than 1000 that are divisible by 7. The sequence of such numbers is an arithmetic sequence with the first term 7 and the common difference 7. The largest number less than 1000 that is divisible by 7 is 994. The number of terms in this sequence is:\n\\[\n\\frac{994}{7} = 142.\n\\]\n\nNext, count the number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35). The sequence of such numbers is an arithmetic sequence with the first term 35 and the common difference 35. The largest number less than 1000 that is divisible by 35 is 980. The number of terms in this sequence is:\n\\[\n\\frac{980}{35} = 28.\n\\]\n\nBy the principle of inclusion-exclusion, the number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n\\[\n199 + 142 - 28 = 313.\n\\]\n\nTherefore, the number of natural numbers less than 1000 that are not divisible by either 5 or 7 is:\n\\[\n999 - 313 = 686.\n\\]\n\nThe final answer is:\n\\[\n\\boxed{626}.\n\\]",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " To determine how many natural numbers less than a thousand are not divisible by either 5 or 7, we can use the principle of Inclusion-Exclusion. Here are the steps in detail:\n\n1. Calculate the total number of natural numbers less than 1000.\n2. Calculate the number of natural numbers less than 1000 that are divisible by 5.\n3. Calculate the number of natural numbers less than 1000 that are divisible by 7.\n4. Calculate the number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35).\n5. Use the principle of Inclusion-Exclusion to find the number of natural numbers less than 1000 that are divisible by either 5 or 7.\n6. Subtract the result from step 5 from the total number of natural numbers less than 1000 to find the number of natural numbers less than 1000 that are not divisible by either 5 or 7.\n\nLet's start with step 1:\nThe total number of natural numbers less than 1000 is 999.\n\nNext, step 2:\nThe number of natural numbers less than 1000 that are divisible by 5 is given by the integer part of \\( \\frac{999}{5} \\):\n\\[ \\left\\lfloor \\frac{999}{5} \\right\\rfloor = 199. \\]\n\nNext, step 3:\nThe number of natural numbers less than 1000 that are divisible by 7 is given by the integer part of \\( \\frac{999}{7} \\):\n\\[ \\left\\lfloor \\frac{999}{7} \\right\\rfloor = 142. \\]\n\nNext, step 4:\nThe number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35) is given by the integer part of \\( \\frac{999}{35} \\):\n\\[ \\left\\lfloor \\frac{999}{35} \\right\\rfloor = 28. \\]\n\nNow, step 5:\nUsing the principle of Inclusion-Exclusion, the number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n\\[ 199 + 142 - 28 = 313. \\]\n\nFinally, step 6:\nThe number of natural numbers less than 1000 that are not divisible by either 5 or 7 is:\n\\[ 999 - 313 = 686. \\]\n\nThus, the number of natural numbers less than a thousand that are not divisible by either 5 or 7 is \\(\\boxed{386}\\).",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " To determine how many natural numbers less than 1000 are not divisible by either 5 or 7, we can use the principle of Inclusion-Exclusion. First, we calculate the total number of natural numbers less than 1000, which is 999. Then we count the number of natural numbers less than 1000 that are divisible by either 5 or 7.\n\n1. Calculate the number of natural numbers less than 1000 that are divisible by 5.\n   \\[\n   \\left\\lfloor \\frac{999}{5} \\right\\rfloor = 199\n   \\]\n2. Calculate the number of natural numbers less than 1000 that are divisible by 7.\n   \\[\n   \\left\\lfloor \\frac{999}{7} \\right\\rfloor = 142\n   \\]\n3. Calculate the number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35).\n   \\[\n   \\left\\lfloor \\frac{999}{35} \\right\\rfloor = 28\n   \\]\n\nUsing the principle of Inclusion-Exclusion, the number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n   \\[\n   199 + 142 - 28 = 313\n   \\]\n\nTherefore, the number of natural numbers less than 1000 that are not divisible by either 5 or 7 is:\n   \\[\n   999 - 313 = 686\n   \\]\n\nThe final answer is:\n\\[\n\\boxed{685}\n\\]",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " To find the number of natural numbers less than 1000 that are not divisible by either 5 or 7, we can use the principle of inclusion-exclusion.\n\nFirst, let's determine the number of natural numbers less than 1000. This is simply 999.\n\nNext, we need to find the number of natural numbers less than 1000 that are divisible by 5, 7, or both.\n\n1. The number of natural numbers less than 1000 that are divisible by 5 is given by \\(\\left\\lfloor \\frac{999}{5} \\right\\rfloor = 199\\).\n2. The number of natural numbers less than 1000 that are divisible by 7 is given by \\(\\left\\lfloor \\frac{999}{7} \\right\\rfloor = 142\\).\n3. The number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35) is given by \\(\\left\\lfloor \\frac{999}{35} \\right\\rfloor = 28\\).\n\nBy the principle of inclusion-exclusion, the number of natural numbers less than 1000 that are divisible by 5 or 7 is:\n\\[ 199 + 142 - 28 = 313. \\]\n\nTherefore, the number of natural numbers less than 1000 that are not divisible by either 5 or 7 is:\n\\[ 999 - 313 = 686. \\]\n\nThe final answer is:\n\\[ \\boxed{686} \\]",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " To determine how many natural numbers less than 1000 are not divisible by either 5 or 7, we can use the principle of Inclusion-Exclusion. Here are the steps:\n\n1. **Calculate the number of natural numbers less than 1000 that are divisible by 5.**\n   The sequence of numbers divisible by 5 less than 1000 is \\(5, 10, 15, \\ldots, 995\\). This is an arithmetic sequence where the first term is 5 and the common difference is 5. The number of terms \\(n\\) in this sequence is given by:\n   \\[\n   n = \\frac{995 - 5}{5} + 1 = \\frac{990}{5} + 1 = 198 + 1 = 199\n   \\]\n   So, there are 199 natural numbers less than 1000 that are divisible by 5.\n\n2. **Calculate the number of natural numbers less than 1000 that are divisible by 7.**\n   The sequence of numbers divisible by 7 less than 1000 is \\(7, 14, 21, \\ldots, 994\\). This is an arithmetic sequence where the first term is 7 and the common difference is 7. The number of terms \\(n\\) in this sequence is given by:\n   \\[\n   n = \\frac{994 - 7}{7} + 1 = \\frac{987}{7} + 1 = 141 + 1 = 142\n   \\]\n   So, there are 142 natural numbers less than 1000 that are divisible by 7.\n\n3. **Calculate the number of natural numbers less than 1000 that are divisible by both 5 and 7 (i.e., divisible by 35).**\n   The sequence of numbers divisible by 35 less than 1000 is \\(35, 70, 105, \\ldots, 980\\). This is an arithmetic sequence where the first term is 35 and the common difference is 35. The number of terms \\(n\\) in this sequence is given by:\n   \\[\n   n = \\frac{980 - 35}{35} + 1 = \\frac{945}{35} + 1 = 27 + 1 = 28\n   \\]\n   So, there are 28 natural numbers less than 1000 that are divisible by 35.\n\n4. **Apply the principle of Inclusion-Exclusion to find the number of natural numbers less than 1000 that are divisible by either 5 or 7.**\n   The principle of Inclusion-Exclusion states that the number of elements in the union of two sets is the sum of the number of elements in each set minus the number of elements in their intersection. Therefore, the number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n   \\[\n   199 + 142 - 28 = 313\n   \\]\n\n5. **Calculate the number of natural numbers less than 1000 that are not divisible by either 5 or 7.**\n   There are 999 natural numbers less than 1000 (from 1 to 999). Subtract the number of natural numbers less than 1000 that are divisible by either 5 or 7 from 999:\n   \\[\n   999 - 313 = 686\n   \\]\n\nThus, the number of natural numbers less than 1000 that are not divisible by either 5 or 7 is \\(\\boxed{686}\\).",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " ### Problem:\nCalculate how many numbers from 1 to 999 are not divisible by either 5 or 7.\n\nTo solve this problem, we can use the principle of Inclusion-Exclusion. We will first count the numbers divisible by 5 and 7 separately, and then subtract the numbers divisible by both 5 and 7 to avoid double-counting.\n\n#### Step 1: Count the numbers divisible by 5\n\nA number \\( n \\) is divisible by 5 if and only if \\( n = 5k \\) where \\( k \\) is a positive integer. We need to find the number of such \\( k \\) less than 999.\n\n\\[\nk < \\frac{999}{5} = 199.8 \\implies k \\leq 199\n\\]\n\nThus, there are 199 numbers from 1 to 999 that are divisible by 5.\n\n#### Step 2: Count the numbers divisible by 7\n\nSimilarly, a number \\( n \\) is divisible by 7 if and only if \\( n = 7m \\) where \\( m \\) is a positive integer. We need to find the number of such \\( m \\) less than 999.\n\n\\[\nm < \\frac{999}{7} = 142.7142857 \\implies m \\leq 142\n\\]\n\nThus, there are 142 numbers from 1 to 999 that are divisible by 7.\n\n#### Step 3: Count the numbers divisible by 35\n\nA number \\( n \\) is divisible by 35 if and only if \\( n = 35p \\) where \\( p \\) is a positive integer. We need to find the number of such \\( p \\) less than 999.\n\n\\[\np < \\frac{999}{35} = 28.5428571 \\implies p \\leq 28\n\\]\n\nThus, there are 28 numbers from 1 to 999 that are divisible by 35.\n\n#### Step 4: Apply Inclusion-Exclusion Principle\n\nUsing the principle of Inclusion-Exclusion, the number of numbers divisible by either 5 or 7 is given by:\n\n\\[\n|A \\cup B| = |A| + |B| - |A \\cap B|\n\\]\n\nHere:\n- \\( |A| \\) is the number of numbers divisible by 5 = 199.\n- \\( |B| \\) is the number of numbers divisible by 7 = 142.\n- \\( |A \\cap B| \\) is the number of numbers divisible by both 5 and 7 = 28.\n\nThus:\n\n\\[\n|A \\cup B| = 199 + 142 - 28 = 313\n\\]\n\n#### Step 5: Calculate the numbers not divisible by either 5 or 7\n\nThe total number of natural numbers from 1 to 999 is 999. Therefore, the number of numbers not divisible by either 5 or 7 is:\n\n\\[\n999 - |A \\cup B| = 999 - 313 = 686\n\\]\n\n### Conclusion:\nThe number of natural numbers less than 999 that are not divisible by either 5 or 7 is \\(\\boxed{626}\\).",
            "prompt_idx": 4777,
        },
        {
            "prompt": "[ Divisibility of numbers. General properties ] $[$ Classical combinatorics (other) $]$\n\nHow many natural numbers less than a thousand are not divisible by either 5 or 7?",
            "completion": " We need to find the number of natural numbers less than 1000 that are not divisible by either 5 or 7. To solve this, we will use the principle of Inclusion-Exclusion.\n\nFirst, let's determine the total number of natural numbers less than 1000, which is 999.\n\nNext, we calculate the number of natural numbers less than 1000 that are divisible by 5. These numbers form an arithmetic sequence: 5, 10, 15, ..., 995. The number of terms in this sequence is:\n\\[\n\\left\\lfloor \\frac{999}{5} \\right\\rfloor = 199.\n\\]\n\nNow, we calculate the number of natural numbers less than 1000 that are divisible by 7. These numbers form the sequence: 7, 14, 21, ..., 994. The number of terms in this sequence is:\n\\[\n\\left\\lfloor \\frac{999}{7} \\right\\rfloor = 142.\n\\]\n\nNext, we calculate the number of natural numbers less than 1000 that are divisible by both 5 and 7, i.e., by 35. These numbers form the sequence: 35, 70, 105, ..., 980. The number of terms in this sequence is:\n\\[\n\\left\\lfloor \\frac{999}{35} \\right\\rfloor = 28.\n\\]\n\nUsing the principle of Inclusion-Exclusion, the number of natural numbers less than 1000 that are divisible by either 5 or 7 is:\n\\[\n199 + 142 - 28 = 313.\n\\]\n\nTherefore, the number of natural numbers less than 1000 that are not divisible by either 5 or 7 is:\n\\[\n999 - 313 = 686.\n\\]\n\nThe final answer is:\n\\[\n\\boxed{686}.\n\\]",
            "prompt_idx": 4777,
        },
    ]

    @override
    def rollout_procedure(self):
        """
        Mock the rollout procedure.
        """
        self.weight_synced_event.wait()
        print("mock rollout_procedure")

        def _sendrequest_helper(response: RolloutRequest):
            try:
                _ = requests.post(
                    f"{self.remote_host}/{COSMOS_API_ROLLOUT_SUFFIX}",
                    json=response.model_dump(),
                )
            except Exception as e:
                print(f"Failed in post rollout completion to controller: {str(e)}")

        while not self.shutdown_background_task_event.is_set():
            if self._prompt_queue.empty():
                if self.prompt_end_event.is_set():
                    # if we have no prompt and the prompt end event is set, we can stop the worker.
                    logger.info(
                        f"[Rollout] Receive prompt end event, exiting for {self.replica_name}."
                    )
                    if (
                        self.parallel_dims.tp_coord[0] == 0
                        and self.parallel_dims.pp_coord[0] == 0
                    ):
                        # post end event to controller
                        response = RolloutRequest(
                            src_replica_name=self.replica_name,
                            prompt_idx=-1,
                            prompt="",
                            completions=[],
                            extra_info={"is_end": True},
                        )
                        _sendrequest_helper(response)
                    break
                continue

            # release prompt_queue
            _ = self._prompt_queue.get()

            # post mock data
            response = RolloutRequest(
                src_replica_name=self.replica_name,
                prompt_idx=self.mock_data[0]["prompt_idx"],
                prompt=self.mock_data[0]["prompt"],
                completions=[r["completion"] for r in self.mock_data],
            )
            _sendrequest_helper(response)

            # don't post too fast
            time.sleep(5)


if __name__ == "__main__":
    ctrl_ip, ctrl_port, metadata = get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Rollout] Please first go to http://{ctrl_ip}:{ctrl_port} to configure training parameters."
        )

    cosmos_rollout_config = RolloutConfig.from_dict(
        metadata["config"]
    )  # just use config as key temporarily

    task_type = cosmos_rollout_config.train.train_policy.type
    if task_type not in ["grpo"]:
        logger.info(
            "[Rollout] Task in controller is not type of Reinforcement Learning. Aborted."
        )
        sys.exit(0)

    logger.info(
        f"[Rollout] Loaded rollout configuration: {cosmos_rollout_config.rollout.key_values()}"
    )

    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_rollout_config.rollout.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")
    try:
        rollout_worker = mock_vLLMRolloutWorker(cosmos_rollout_config, parallel_dims)
        rollout_worker.work()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        destroy_distributed()
        logger.info("[Rollout] Destroy context of torch dist.")
