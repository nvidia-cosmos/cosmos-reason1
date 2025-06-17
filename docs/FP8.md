## FP8 Training


Refered to [torchtitan](https://github.com/pytorch/torchtitan), we support `tensorwise` and `rowwise` FP8 quantization in training now.

User coud add fp8 configs into the toml file to enable fp8. FP8Config has fields:


```
enable_fp8: Whether to enable FP8.
fp8_recipe: dynamic_scaling or delayed_scaling, time to collect the scale factors.
quant_recipe: `tensorwise` and `rowwise` supported now.
```

Example:
```
[train.fp8]
enable_fp8 = true
fp8_recipe = "dynamic_scaling"
quant_recipe = "rowwise"
```

