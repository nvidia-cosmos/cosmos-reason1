# VLADBench

Post-training example using VLADBench dataset: https://huggingface.co/datasets/depth2world/VLADBench

[Setup the repository](../../docs/UserGuide.md#Setup)

Download the dataset:

```shell
python examples/nexar_collision_prediction/download.py data/nexar_collision_prediction
```

Post-train the model:

```sh
cosmos-rl --config examples/nexar_collision_prediction/sft.toml ./tools/dataset/cosmos_sft.py
```
