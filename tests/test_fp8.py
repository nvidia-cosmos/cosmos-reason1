import pytest

from torch import nn


from cosmos_reason1.utils.fp8.fp8_util import (
    FP8ModelConverter,
    IS_TORCH_COMPATIBLE_WITH_FP8,
)
from cosmos_reason1.policy.config import Config as CosmosConfig
from cosmos_reason1.policy.config import TrainingConfig, FP8Config
from cosmos_reason1.utils.parallelism import ParallelDims

from torchao.float8.float8_linear import Float8Linear


class DemoModel(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.linear(self.norm(x)))

    def fqn_filter_for_fp8(self):
        return []


@pytest.mark.parametrize("quant_recipe", ["rowwise", "tensorwise"])
@pytest.mark.parametrize("dp_shard", [1, 2])
def test_fp8_model_converter_delayed_scaling(quant_recipe, dp_shard):
    if not IS_TORCH_COMPATIBLE_WITH_FP8:
        pytest.skip("FP8 is not supported for this version of PyTorch")

    demo_model = DemoModel(input_dim=1024, output_dim=512, intermediate_dim=1024)
    config = CosmosConfig(
        train=TrainingConfig(
            fp8=FP8Config(
                enable_fp8=True, quant_recipe=quant_recipe, fp8_recipe="delayed_scaling"
            )
        )
    )
    # Mock the parallel_dims
    tp_size = 2
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp_shard,
        cp=1,
        tp=tp_size,
        pp=1,
        world_size=dp_shard * tp_size,
        pp_dynamic_shape=False,
    )
    with pytest.raises(NotImplementedError):
        converter = FP8ModelConverter(config, parallel_dims)
        converter.convert_model(demo_model)


@pytest.mark.parametrize("quant_recipe", ["rowwise", "tensorwise"])
@pytest.mark.parametrize("fp8_recipe", ["dynamic_scaling"])
@pytest.mark.parametrize("dp_shard", [1, 2])
def test_fp8_model_converter(quant_recipe, fp8_recipe, dp_shard):
    if not IS_TORCH_COMPATIBLE_WITH_FP8:
        pytest.skip("FP8 is not supported for this version of PyTorch")

    demo_model = DemoModel(input_dim=1024, output_dim=512, intermediate_dim=1024)
    config = CosmosConfig(
        train=TrainingConfig(
            fp8=FP8Config(
                enable_fp8=True, quant_recipe=quant_recipe, fp8_recipe=fp8_recipe
            )
        )
    )
    # Mock the parallel_dims
    tp_size = 2
    parallel_dims = ParallelDims(
        dp_replicate=1,
        dp_shard=dp_shard,
        cp=1,
        tp=tp_size,
        pp=1,
        world_size=dp_shard * tp_size,
        pp_dynamic_shape=False,
    )
    config.train.fp8.quant_recipe = quant_recipe
    config.train.fp8.fp8_recipe = fp8_recipe

    try:
        converter = FP8ModelConverter(config, parallel_dims)
        converter.convert_model(demo_model)
    except Exception as e:
        pytest.fail(f"Failed to Pass FP8 conversion: {e}")

    # Check the model is converted
    assert isinstance(
        demo_model.linear, Float8Linear
    ), f"Got type: {type(demo_model.linear)}"
    assert isinstance(
        demo_model.linear2, Float8Linear
    ), f"Got type: {type(demo_model.linear2)}"
