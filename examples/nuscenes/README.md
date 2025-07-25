# nuScenes

This belongs in the playbook.

1. Normalize dataset format.
1. Run cosmos-curate split/shard.
1. [Optional] Download captions from S3, pass to LLM, upload back to S3.
1. Download webdataset and convert to huggingface arrow.
    1. Ideally, fix qwen-vl-utils to read `io.Bytes` directly to skip this step. This is required for multi-node.

```sh
# Download cosmos-curate output
aws s3 cp --recursive "s3://lha-datasets/cosmos-reason1/cosmos-curate/v0" data/webdataset/webdataset

python tools/dataset/process_webdataset.py data/webdataset/webdataset data/webdataset/raw
python examples/nuscenes/process.py data/webdataset/raw data/webdataset/processed

cosmos-rl --config examples/nuscenes/sft.toml ./tools/dataset/cosmos_sft.py
```
