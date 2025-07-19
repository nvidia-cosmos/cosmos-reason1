"""Download Nexar collision prediction dataset."""

import argparse
import json
import datasets
from rich import print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    dataset = datasets.load_dataset(
        "nexar-ai/nexar_collision_prediction", split="train"
    )
    print(dataset)
    dataset = dataset.cast_column("video", datasets.Video(decode=False))

    def process_sample(sample):
        conversations = [
            {
                "role": "system",
                "content": "Answer the questions.",
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
                        "text": "What is the weather in this video? Choose from ['Rain', 'Cloudy', 'Snow', 'Clear'].",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": str(sample["weather"]),
            },
        ]
        return {
            "conversations": json.dumps(conversations),
            "video": sample["video"],
        }

    dataset = list(map(process_sample, dataset))
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(args.output)


if __name__ == "__main__":
    main()
