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

from cosmos_reason1.utils.parallelism import ParallelDims, ParallelismConfig
from typing import Dict, List, Tuple, Callable, Any
import torch


class DimRankInfo:
    rank: int
    size: int
    dim: str
    length: int = 1

    def __init__(self, rank: int, size: int, dim: str, length: int = 1):
        """
        Initialize the DimRankInfo with the given rank, size, and dimension.
        """
        self.rank = rank
        self.size = size
        self.dim = dim
        self.length = length

    def __repr__(self):
        # Returning a dictionary representation
        return f"{self.__dict__}"


def slice_tensor_with_strategy(
    tensor: torch.Tensor, idx: int, tensor_split_strategy: DimRankInfo
):
    view = tensor
    assert view.shape[idx] % tensor_split_strategy.size == 0
    start = view.shape[idx] // tensor_split_strategy.size * tensor_split_strategy.rank
    length = (
        view.shape[idx] // tensor_split_strategy.size * tensor_split_strategy.length
    )
    if view.dim() == 1:
        if idx == 0:
            view = view[start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 1D tensor.")
    elif view.dim() == 2:
        if idx == 0:
            view = view[start : start + length, :]
        elif idx == 1:
            view = view[:, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 2D tensor.")
    elif view.dim() == 3:
        if idx == 0:
            view = view[start : start + length, :, :]
        elif idx == 1:
            view = view[:, start : start + length, :]
        elif idx == 2:
            view = view[:, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 3D tensor.")
    elif view.dim() == 4:
        if idx == 0:
            view = view[start : start + length, :, :, :]
        elif idx == 3:
            view = view[:, :, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 4D tensor.")
    elif view.dim() == 5:
        if idx == 0:
            view = view[start : start + length, :, :, :, :]
        elif idx == 4:
            view = view[:, :, :, :, start : start + length]
        else:
            raise ValueError(f"Invalid index {idx} for 5D tensor.")
    else:
        raise ValueError(f"Invalid tensor dimension {view.dim()}.")
    return view


def slice_tensor_with_strategies(
    tensor: torch.Tensor, strategys: Dict[int, DimRankInfo]
) -> torch.Tensor:
    """
    Slices the tensor according to the given strategies.
    :param tensor: The tensor to be sliced.
    :param strategys: A dictionary mapping dimension indices to DimRankInfo objects.
    :return: The sliced tensor.
    """
    view = tensor
    for idx, split in strategys.items():
        view = slice_tensor_with_strategy(view, idx, split)
    return view


class ParallelTopoMapper:
    """
    A class to represent a weight sharing topology map for weight synchronization.
    """

    policy_parallelism: ParallelDims
    rollout_parallelism: ParallelDims
    ordered_dims: List[str] = ["tp", "dp_shard_cp"]

    def __init__(
        self,
        policy_parallelism: ParallelismConfig,
        rollout_parallelism: ParallelismConfig,
        poilcy_world_size: int,
        rollout_world_size: int,
        policy_parallelism_strategy: Callable,
        rollout_parallelism_strategy: Callable,
        model_config: Any,
    ):
        """
        Initialize the ParallelTopoMap with the given parallelism configurations.

        :param policy_parallelism: The parallelism configuration for the policy.
        :param rollout_parallelism: The parallelism configuration for the rollout.
        :param poilcy_world_size: The world size for the policy.
        :param rollout_world_size: The world size for the rollout.
        :param policy_parallelism_strategy: The strategy for the policy parallelism.
        :param rollout_parallelism_strategy: The strategy for the rollout parallelism.
        :param model_config: The model configuration.
        """
        self.policy_parallelism = ParallelDims.from_config_for_analysis(
            policy_parallelism, poilcy_world_size
        )
        self.policy_parallelism.build_mesh_info()

        self.rollout_parallelism = ParallelDims.from_config_for_analysis(
            rollout_parallelism, rollout_world_size
        )
        self.rollout_parallelism.build_mesh_info()
        self.policy_parallelism_strategy = policy_parallelism_strategy
        self.rollout_parallelism_strategy = rollout_parallelism_strategy
        self.calculate_sharing_map()
        self.model_config = model_config

    def get_full_rank_in_policy(self, global_rank: int) -> Dict[str, DimRankInfo]:
        """
        Get the full rank of the given rank in the simulation map.

        :param rank: The rank to get the full rank for.
        :return: A dictionary mapping each parallel dimension to its full rank.
        """
        full_rank = {}
        for dim in self.ordered_dims:
            full_rank[dim] = DimRankInfo(
                self.policy_parallelism.get_rank_in_dim(dim, global_rank),
                self.policy_parallelism.get_size_in_dim(dim),
                dim,
            )
        full_rank["pp"] = DimRankInfo(
            self.policy_parallelism.get_rank_in_dim("pp", global_rank),
            self.policy_parallelism.get_size_in_dim("pp"),
            "pp",
        )
        return full_rank

    def get_full_rank_in_rollout(self, global_rank: int) -> Dict[str, DimRankInfo]:
        """
        Get the full rank of the given rank in the simulation map.

        :param rank: The rank to get the full rank for.
        :return: A dictionary mapping each parallel dimension to its full rank.
        """
        full_rank = {}
        for dim in self.ordered_dims:
            full_rank[dim] = DimRankInfo(
                self.rollout_parallelism.get_rank_in_dim(dim, global_rank),
                self.rollout_parallelism.get_size_in_dim(dim),
                dim,
            )
        full_rank["pp"] = DimRankInfo(
            self.rollout_parallelism.get_rank_in_dim("pp", global_rank),
            self.rollout_parallelism.get_size_in_dim("pp"),
            "pp",
        )
        return full_rank

    def get_unified_rank_info(
        self, a: DimRankInfo, b: DimRankInfo
    ) -> Tuple[DimRankInfo, DimRankInfo]:
        """
        Get the unified rank information for two DimRankInfo objects.
        :param a: The first DimRankInfo object.
        :param b: The second DimRankInfo object.
        :return: A tuple containing the unified rank information for both objects.
        """
        size = max(a.size, b.size)
        assert (
            size % a.size == 0 and size % b.size == 0
        ), "Sizes are not compatible for unification"
        scale_a = size // a.size
        scale_b = size // b.size
        scaled_a_size = a.size * scale_a
        scaled_b_size = b.size * scale_b
        scaled_a_rank = a.rank * scale_a
        scaled_b_rank = b.rank * scale_b
        unified_a = DimRankInfo(scaled_a_rank, scaled_a_size, a.dim, a.length * scale_a)
        unified_b = DimRankInfo(scaled_b_rank, scaled_b_size, b.dim, b.length * scale_b)
        return unified_a, unified_b

    def rank_overlap(self, a: DimRankInfo, b: DimRankInfo) -> DimRankInfo:
        """
        Check if the ranks of two DimRankInfo objects overlap.

        :param a: The first DimRankInfo object.
        :param b: The second DimRankInfo object.
        :return: A DimRankInfo object representing the overlap, or None if there is no overlap.
        """
        assert a.dim == b.dim, "Dimensions do not match"
        a_new, b_new = self.get_unified_rank_info(a, b)
        assert a_new.size == b_new.size, "Sizes do not match after unification"
        left = max(a_new.rank, b_new.rank)
        right = min(
            a_new.rank + a_new.length,
            b_new.rank + b_new.length,
        )
        overlapped = None
        if left < right:
            overlapped = DimRankInfo(left, a_new.size, a_new.dim, right - left)
        return overlapped

    def relative_rank(self, smaller: DimRankInfo, larger: DimRankInfo) -> DimRankInfo:
        """
        Get the relative rank of two DimRankInfo objects.
        :param smaller: The smaller DimRankInfo object.
        :param larger: The larger DimRankInfo object.
        :return: A DimRankInfo object representing the relative rank.
        """
        s, l = self.get_unified_rank_info(smaller, larger)  # noqa: E741
        assert s.rank >= l.rank, "Smaller rank is not less than or equal to larger rank"
        assert (
            s.rank + s.length <= l.rank + l.length
        ), "Smaller rank does not fit within larger rank"
        rank = s.rank - l.rank
        size = l.length
        length = s.length
        return DimRankInfo(rank, size, s.dim, length)

    def merge_rank(self, outter: DimRankInfo, inner: DimRankInfo) -> DimRankInfo:
        """
        Merge two DimRankInfo objects into one.
        :param outter: The outer DimRankInfo object.
        :param inner: The inner DimRankInfo object.
        :return: A DimRankInfo object representing the merged rank.
        """
        assert outter.length == 1, "Outer rank length must be 1"
        size = outter.size * inner.size
        rank = outter.rank * inner.size + inner.rank
        length = inner.length
        return DimRankInfo(rank, size, outter.dim, length)

    def check_ranks_overlap(
        self, a: Dict[str, DimRankInfo], b: Dict[str, DimRankInfo]
    ) -> bool:
        """
        Check if the ranks in the two dictionaries overlap.

        :param a: The first dictionary.
        :param b: The second dictionary.
        :return: True if there is an overlap, False otherwise.
        """
        for dim in self.ordered_dims:
            if self.rank_overlap(a[dim], b[dim]) is None:
                return False
        return True

    def calculate_sharing_map(
        self,
    ) -> Tuple[
        List[List[bool]], List[Dict[str, DimRankInfo]], List[Dict[str, DimRankInfo]]
    ]:
        """
        Calculate the sharing map based on the parallelism configurations.
        """
        policy_ranks = range(self.policy_parallelism.world_size)
        rollout_ranks = range(self.rollout_parallelism.world_size)
        policy_dp_shard_cp = (
            self.policy_parallelism.dp_shard * self.policy_parallelism.cp
        )
        rollout_dp_shard_cp = (
            self.rollout_parallelism.dp_shard * self.rollout_parallelism.cp
        )
        policy_tp_size = self.policy_parallelism.tp
        rollout_tp_size = self.rollout_parallelism.tp

        sharing_map = [[False for _ in rollout_ranks] for _ in policy_ranks]
        policy_full_rank_map = []
        rollout_full_rank_map = []

        for p in policy_ranks:
            policy_full_rank = self.get_full_rank_in_policy(p)
            policy_full_rank_map.append(policy_full_rank)
            for r in rollout_ranks:
                rollout_full_rank = self.get_full_rank_in_rollout(r)
                if p == 0:
                    rollout_full_rank_map.append(rollout_full_rank)
                assert (
                    policy_full_rank["dp_shard_cp"].rank < policy_dp_shard_cp
                ), "Policy dp_shard_cp rank out of range"
                assert (
                    rollout_full_rank["dp_shard_cp"].rank < rollout_dp_shard_cp
                ), "Rollout dp_shard_cp rank out of range"
                assert (
                    policy_full_rank["tp"].rank < policy_tp_size
                ), "Policy tp rank out of range"
                assert (
                    rollout_full_rank["tp"].rank < rollout_tp_size
                ), "Rollout tp rank out of range"
                overlap = self.check_ranks_overlap(policy_full_rank, rollout_full_rank)
                sharing_map[p][r] = overlap
        self.policy_full_rank_map = policy_full_rank_map
        self.rollout_full_rank_map = rollout_full_rank_map
        self.sharing_map = sharing_map

    def dim_tensor_sharing_info(
        self,
        policy_rank: Dict[str, DimRankInfo],
        rollout_rank: Dict[str, DimRankInfo],
        dim: str,
    ) -> Tuple[DimRankInfo, DimRankInfo]:
        """
        Get the tensor sharing information one different dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A set of dimensions that are different.
        """
        if dim not in policy_rank:
            p = DimRankInfo(0, 1, dim)
        else:
            p = policy_rank[dim]
        if dim not in rollout_rank:
            r = DimRankInfo(0, 1, dim)
        else:
            r = rollout_rank[dim]

        p_new, r_new = self.get_unified_rank_info(p, r)
        overlap = self.rank_overlap(p_new, r_new)
        if overlap is None:
            # logger.warning(f"No rank overlap detected in {dim}: {p} and {r}")
            return None, None
        overlap_r = self.relative_rank(overlap, r_new)
        overlap_p = self.relative_rank(overlap, p_new)
        return overlap_p, overlap_r

    def diff_dim_tensor_sharing_info(
        self, policy_rank: Dict[str, DimRankInfo], rollout_rank: Dict[str, DimRankInfo]
    ) -> Tuple[Dict[str, DimRankInfo], Dict[str, DimRankInfo]]:
        """
        Get the tensor sharing information one different dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A set of dimensions that are different.
        """
        overlap_in_policy = {}
        overlap_in_rollout = {}
        for dim in self.ordered_dims:
            overlap_p, overlap_r = self.dim_tensor_sharing_info(
                policy_rank, rollout_rank, dim
            )
            overlap_in_policy[dim] = overlap_p
            overlap_in_rollout[dim] = overlap_r
        return overlap_in_policy, overlap_in_rollout

    def merged_dim_tensor_sharing_info(
        self, policy_rank: Dict[str, DimRankInfo], rollout_rank: Dict[str, DimRankInfo]
    ) -> Tuple[DimRankInfo, DimRankInfo]:
        """
        Get the tensor sharing information for the same dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A set of dimensions that are the same.
        """
        assert len(self.ordered_dims) == 2, "Only two dimensions are supported"
        if self.ordered_dims[0] not in policy_rank:
            policy_rank[self.ordered_dims[0]] = DimRankInfo(0, 1, self.ordered_dims[0])
        if self.ordered_dims[1] not in policy_rank:
            policy_rank[self.ordered_dims[1]] = DimRankInfo(0, 1, self.ordered_dims[1])
        if self.ordered_dims[0] not in rollout_rank:
            rollout_rank[self.ordered_dims[0]] = DimRankInfo(0, 1, self.ordered_dims[0])
        if self.ordered_dims[1] not in rollout_rank:
            rollout_rank[self.ordered_dims[1]] = DimRankInfo(0, 1, self.ordered_dims[1])
        # Merge the ranks of the two dimensions
        p = self.merge_rank(
            policy_rank[self.ordered_dims[0]], policy_rank[self.ordered_dims[1]]
        )
        r = self.merge_rank(
            rollout_rank[self.ordered_dims[0]], rollout_rank[self.ordered_dims[1]]
        )
        overlap = self.rank_overlap(p, r)
        if overlap is None:
            # logger.warning(
            #     f"No rank overlap detected in {self.ordered_dims[0]} and {self.ordered_dims[1]}: {p} and {r}"
            # )
            return None, None
        overlap_r = self.relative_rank(overlap, r)
        overlap_p = self.relative_rank(overlap, p)
        return overlap_p, overlap_r

    @classmethod
    def get_global_ranks_for_given_group_rank(
        cls, parallel_dims: ParallelDims, group_rank: Dict[str, int]
    ) -> List[int]:
        """
        Get the global ranks for a given group rank in the parallelism configuration.
        group_rank is subset of parallel_dims, so there could be multiple devices have
        the same group_rank.
        :param parallel_dims: The parallelism configuration.
        :param group_rank: The group rank to get the global ranks for.
        :return: A list of global ranks.
        """
        if len(group_rank) == 0:
            return list(range(parallel_dims.world_size))
        global_ranks = []
        for rank in range(parallel_dims.world_size):
            if all(
                [
                    parallel_dims.get_rank_in_dim(dim, rank) == dimr
                    for dim, dimr in group_rank.items()
                ]
            ):
                global_ranks.append(rank)
        return global_ranks

    def policy_duplicate_ranks_at_given_dimensions(
        self, dims: List[str], global_rank: int
    ) -> List[int]:
        """
        Get the policy duplicate ranks at the given dimensions.
        :param dims: The dimensions to check.
        :param global_rank: The global rank to check.
        :return: A list of duplicate ranks.
        """
        dims_map = {}
        for dim in dims:
            dims_map[dim] = self.policy_parallelism.get_rank_in_dim(dim, global_rank)

        return ParallelTopoMapper.get_global_ranks_for_given_group_rank(
            self.policy_parallelism, dims_map
        )

    def rollout_duplicate_ranks_at_given_dimensions(
        self, dims: List[str], global_rank: int
    ) -> List[int]:
        """
        Get the rollout duplicate ranks at the given dimensions.
        :param dims: The dimensions to check.
        :param global_rank: The global rank to check.
        :return: A list of duplicate ranks.
        """
        dims_map = {}
        for dim in dims:
            dims_map[dim] = self.rollout_parallelism.get_rank_in_dim(dim, global_rank)

        return ParallelTopoMapper.get_global_ranks_for_given_group_rank(
            self.rollout_parallelism, dims_map
        )

    def generate_slice_strategies(
        self,
        dim_to_parallel: Dict[int, list[str]],
        policy_rank: Dict[str, DimRankInfo],
        rollout_rank: Dict[str, DimRankInfo],
    ) -> Tuple[Dict[int, DimRankInfo], Dict[int, DimRankInfo]]:
        """
        Generate slice strategies for the given dimensions and ranks.
        :param dim_to_parallel: A dictionary mapping dimension indices to parallel dimensions.
        :param policy_rank: The policy rank information.
        :param rollout_rank: The rollout rank information.
        :return: A dictionary mapping dimension indices to DimRankInfo objects.
        """
        p_tensor_split_strategys = {}
        r_tensor_split_strategys = {}
        for idx, dims in dim_to_parallel.items():
            if len(dims) == 1:
                p_tensor_split_strategy, r_tensor_split_strategy = (
                    self.dim_tensor_sharing_info(policy_rank, rollout_rank, dims[0])
                )
                if p_tensor_split_strategy is None:
                    assert r_tensor_split_strategy is None
                    return None, None
            elif len(dims) == 2:
                p_tensor_split_strategy, r_tensor_split_strategy = (
                    self.merged_dim_tensor_sharing_info(policy_rank, rollout_rank)
                )
                if p_tensor_split_strategy is None:
                    assert r_tensor_split_strategy is None
                    return None, None
            else:
                raise ValueError(
                    f"Invalid dimension mapping: {dims} in generate_slice_strategies"
                )
            p_tensor_split_strategys[idx] = p_tensor_split_strategy
            r_tensor_split_strategys[idx] = r_tensor_split_strategy
        return p_tensor_split_strategys, r_tensor_split_strategys

    def policy_to_rollout_assign(
        self, policys: List[int], rollouts: List[int]
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Assign policy ranks to rollout ranks based on the sharing map.
        :param policys: The list of policy ranks.
        :param rollouts: The list of rollout ranks.
        :return: A tuple containing two dictionaries: policy assignment and rollout assignment.
        """
        p_assignment = {}
        r_assignment = {}
        if len(policys) >= len(rollouts):
            for i, p in enumerate(policys):
                if i >= len(rollouts):
                    break
                p_assignment[p] = [rollouts[i]]
                r_assignment[rollouts[i]] = [p]
        else:
            group_size = ((len(rollouts) - 1) // len(policys)) + 1
            for i, p in enumerate(policys):
                rs = rollouts[
                    i * group_size : min(i * group_size + group_size, len(rollouts))
                ]
                p_assignment[p] = rs
                for r in rs:
                    if r not in r_assignment:
                        r_assignment[r] = []
                    r_assignment[r].append(p)
        for p in policys:
            if p not in p_assignment:
                p_assignment[p] = []
        for r in rollouts:
            if r not in r_assignment:
                r_assignment[r] = []
        return p_assignment, r_assignment

    @classmethod
    def merge_dim_to_parallel(
        cls,
        dim_to_parallel: Dict[int, list[str]],
        dim_to_parallel_2: Dict[int, list[str]],
    ) -> Dict[int, list[str]]:
        """
        Merge two dictionaries of dimension to parallel dimensions.
        :param dim_to_parallel: The first dictionary.
        :param dim_to_parallel_2: The second dictionary.
        :return: A merged dictionary.
        """
        merged = {}
        for idx, dims in dim_to_parallel.items():
            if idx not in merged:
                merged[idx] = []
            for d in dims:
                if d not in merged[idx]:
                    merged[idx].append(d)
        for idx, dims in dim_to_parallel_2.items():
            if idx not in merged:
                merged[idx] = []
            for d in dims:
                if d not in merged[idx]:
                    merged[idx].append(d)
        return merged

    def generate_policy_to_rollout_insts(
        self,
        params: List[Tuple[str, Tuple[int]]],
        global_rank: int,
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        """
        Generate policy to rollout broadcast instructions.
        :param params: The parameters to generate instructions for.
        :param global_rank: The global rank to generate instructions for.
        :return: A list of tuples containing the generated instructions.
        Tuple meaning: [sender rank, receiver rank, tensor split strategies, tensor name, tensor shape]
        """
        policy_to_rollput_insts = []
        for dest_name, shape in params:
            p_split_dim_map, p_dim_to_parallel, p_pp_rank = (
                self.policy_parallelism_strategy(
                    shape, dest_name, self.policy_parallelism, self.model_config
                )
            )
            r_split_dim_map, r_dim_to_parallel, r_pp_rank = (
                self.rollout_parallelism_strategy(
                    shape, dest_name, self.rollout_parallelism, self.model_config
                )
            )
            p_rank = global_rank
            p_dup_ranks = self.policy_duplicate_ranks_at_given_dimensions(
                list(p_split_dim_map.keys()) + ["pp"], p_rank
            )
            dim_to_parallel = ParallelTopoMapper.merge_dim_to_parallel(
                p_dim_to_parallel, r_dim_to_parallel
            )
            p_ranks = self.policy_full_rank_map[p_rank]
            if p_ranks["pp"].rank != p_pp_rank:
                continue
            for r_rank in range(self.rollout_parallelism.world_size):
                r_dup_ranks = self.rollout_duplicate_ranks_at_given_dimensions(
                    list(r_split_dim_map.keys()) + ["pp"], r_rank
                )
                r_ranks = self.rollout_full_rank_map[r_rank]
                if r_ranks["pp"].rank != r_pp_rank:
                    continue
                tensor_split_strategys, r_tensors = self.generate_slice_strategies(
                    dim_to_parallel, p_ranks, r_ranks
                )
                if tensor_split_strategys is None:
                    continue
                else:
                    assignments, _ = self.policy_to_rollout_assign(
                        p_dup_ranks, r_dup_ranks
                    )
                    assignment = assignments[p_rank]
                    for r in assignment:
                        if r == r_rank:
                            policy_to_rollput_insts.append(
                                (p_rank, r, tensor_split_strategys, dest_name, shape)
                            )
        return policy_to_rollput_insts

    def generate_rollout_from_policy_insts(
        self,
        params: List[Tuple[str, Tuple[int]]],
        rollout_rank: int,
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        """
        Generate rollout from policy broadcast instructions.
        :param params: The parameters to generate instructions for.
        :param rollout_rank: The rollout rank to generate instructions for.
        :return: A list of tuples containing the generated instructions.
        Tuple meaning: [sender rank, receiver rank, tensor split strategies, tensor name, tensor shape]
        """
        rollput_from_policy_insts = []
        for dest_name, shape in params:
            p_split_dim_map, p_dim_to_parallel, p_pp_rank = (
                self.policy_parallelism_strategy(
                    shape, dest_name, self.policy_parallelism, self.model_config
                )
            )
            r_split_dim_map, r_dim_to_parallel, r_pp_rank = (
                self.rollout_parallelism_strategy(
                    shape, dest_name, self.rollout_parallelism, self.model_config
                )
            )
            dim_to_parallel = ParallelTopoMapper.merge_dim_to_parallel(
                p_dim_to_parallel, r_dim_to_parallel
            )
            r_rank = rollout_rank
            r_dup_ranks = self.rollout_duplicate_ranks_at_given_dimensions(
                list(r_split_dim_map.keys()) + ["pp"], r_rank
            )
            r_ranks = self.rollout_full_rank_map[r_rank]
            if r_ranks["pp"].rank != r_pp_rank:
                continue
            for p_rank in range(self.policy_parallelism.world_size):
                p_dup_ranks = self.policy_duplicate_ranks_at_given_dimensions(
                    list(p_split_dim_map.keys()) + ["pp"], p_rank
                )
                p_ranks = self.policy_full_rank_map[p_rank]
                if p_ranks["pp"].rank != p_pp_rank:
                    continue
                p_tensors, tensor_split_strategys = self.generate_slice_strategies(
                    dim_to_parallel, p_ranks, r_ranks
                )
                if tensor_split_strategys is None:
                    # print("Tensor split strategies is None")
                    continue
                else:
                    _, assignments = self.policy_to_rollout_assign(
                        p_dup_ranks, r_dup_ranks
                    )
                    assignment = assignments[r_rank]
                    for p in assignment:
                        if p == p_rank:
                            rollput_from_policy_insts.append(
                                (p, r_rank, tensor_split_strategys, dest_name, shape)
                            )
        return rollput_from_policy_insts


class ParallelTopoMapperGroup:
    """
    A class to represent a group of weight sharing topology maps for weight synchronization.
    """

    mapper_group: List[ParallelTopoMapper] = []

    def __init__(
        self,
        policy_parallelism: ParallelismConfig,
        rollout_parallelism: ParallelismConfig,
        poilcy_world_size: int,
        rollout_world_size: int,
        model_config: Any,
        model_path: str,
    ):
        # policy_parallelism_strategy: List[Callable],
        # rollout_parallelism_strategy: List[Callable],
        # ):
        """
        Initialize the ParallelTopoMapperGroup with the given parallelism configurations.

        :param policy_parallelism: The parallelism configuration for the policy.
        :param rollout_parallelism: The parallelism configuration for the rollout.
        :param poilcy_world_size: The world size for the policy.
        :param rollout_world_size: The world size for the rollout.
        :param model_config: The model configuration.
        :param model_path: The path to the model.
        """
        connverted = {
            "gpt": "qwen2",
        }
        self.model_config = model_config
        model_type = model_config.model_type
        if model_type in connverted:
            model_type = connverted[model_type]

        from cosmos_reason1.rollout.weight_mapper.registry import get_weight_mapper

        self.weight_mapper = get_weight_mapper(model_type)(model_path)
        model_num = {"qwen2_5_vl": 2}

        policy_config = self.weight_mapper.get_policy_parallelism(policy_parallelism)
        rollout_config = self.weight_mapper.get_rollout_parallelism(rollout_parallelism)

        # Note: policy_strategies and rollout_strategies callable to decide if or how to parallel
        # the param tensor of a give name.
        policy_strategies = self.weight_mapper.get_policy_parallelism_strategy()
        rollout_strategies = self.weight_mapper.get_rollout_parallelism_strategy()
        n_model = model_num.get(model_type, 1)
        for i in range(n_model):
            # logger.info(
            #     f"Policy parallelism config: {policy_config[i]}, Rollout parallelism config: {rollout_config[i]}"
            #     f"Policy parallelism strategy: {policy_strategies[i]}, Rollout parallelism strategy: {rollout_strategies[i]}"
            # )
            self.mapper_group.append(
                ParallelTopoMapper(
                    policy_config[i],
                    rollout_config[i],
                    poilcy_world_size,
                    rollout_world_size,
                    policy_strategies[i],
                    rollout_strategies[i],
                    model_config,
                )
            )

    def generate_policy_to_rollout_insts_for_index(
        self,
        params: List[Tuple[str, Tuple[int]]],
        global_rank: int,
        index: int = 0,
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        return self.mapper_group[index].generate_policy_to_rollout_insts(
            params, global_rank
        )

    def generate_rollout_from_policy_insts_for_index(
        self, params: List[Tuple[str, Tuple[int]]], rollout_rank: int, index: int
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        return self.mapper_group[index].generate_rollout_from_policy_insts(
            params, rollout_rank
        )

    def resort_params(self, params: List[Tuple[str, Tuple[int]]]):
        """
        Resort the parameters based on the name mapper.
        :param params: The parameters to resort.
        :return: A list of tuples containing the resorted parameters.
        """
        if len(self.mapper_group) == 1:
            return [params]
        new_params = [[] for _ in self.mapper_group]
        for name, shape in params:
            idx = self.weight_mapper.name_to_model_index(name)
            new_params[idx].append((name, shape))
        return new_params

    def generate_policy_to_rollout_insts(
        self,
        params: List[Tuple[str, Tuple[int]]],
        global_rank: int,
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        new_params = self.resort_params(params)
        insts = []
        for index, p in enumerate(new_params):
            insts.extend(
                self.mapper_group[index].generate_policy_to_rollout_insts(
                    p, global_rank
                )
            )
        return insts

    def generate_rollout_from_policy_insts(
        self, params: List[Tuple[str, Tuple[int]]], rollout_rank: int
    ) -> List[Tuple[int, int, Dict[int, DimRankInfo], str, Tuple[int]]]:
        new_params = self.resort_params(params)
        insts = []
        for index, p in enumerate(new_params):
            insts.extend(
                self.mapper_group[index].generate_rollout_from_policy_insts(
                    p, rollout_rank
                )
            )
        return insts
