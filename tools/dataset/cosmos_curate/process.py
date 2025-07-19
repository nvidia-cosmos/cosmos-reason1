"""Process cosmos-curate huggingface dataset."""

import argparse
import json
from pathlib import Path
import datasets
from rich import print

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Input directory.")
    parser.add_argument("output", type=str, help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_from_disk(args.input)

    def process_sample(sample: dict) -> dict:
        key = sample["__key__"]

        metas = json.loads(sample["metas"])
        caption = metas["windows"][0]["qwen_caption"]
        # TODO: Split question/answer
        question = caption
        answer = ""

        conversations = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Please answer the questions.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "video",
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        return {
            "__key__": key,
            "conversations": json.dumps(conversations),
            "video": sample["video"],
        }

    dataset = list(map(process_sample, dataset))
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
