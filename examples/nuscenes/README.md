# nuScenes

Post-training example using [nuScenes](https://www.nuscenes.org/nuscenes).

```sh
# Download dataset
s5cmd cp --show-progress "s3://lha-datasets/users/joallen/nuscenes/v0/shard/v0/*" ~/datasets/nuscenes/v0/shard/
python tools/dataset/cosmos_curate/extract.py ~/datasets/nuscenes/v0/shard ~/datasets/nuscenes/v0/hf
python tools/dataset/cosmos_curate/process.py ~/datasets/nuscenes/v0/hf ~/datasets/nuscenes/v0/cosmos-rl

# Run SFT
cosmos-rl --config examples/nuscenes/sft.toml ./tools/dataset/hf_cosmos_sft.py
```
