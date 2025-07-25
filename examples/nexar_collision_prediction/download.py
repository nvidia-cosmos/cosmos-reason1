"""Download Nexar collision prediction dataset."""

import argparse
import json
from datasets import Video, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    dataset = load_dataset("nexar-ai/nexar_collision_prediction", split="train")
    print(dataset)
    breakpoint()
    dataset = dataset.cast_column("video", Video(decode=False))

    def add_conversations(sample):
        conversations = [
            {
                    "role": "system",
                    "content": "Answer the questions.",
                },
                {
                    "role": "user",
                    "content": json.dumps([
                        {
                            "type": "video",
                            "video": "video",
                        },
                        {
                            "type": "text",
                            "text": "What is the weather in this video? Choose from ['Rain', 'Cloudy', 'Snow', 'Clear'].",
                        },
                    ]),
                },
                {
                    "role": "assistant",
                    "content": str(sample["weather"]),
                },
        ]
        return {
            "conversations": conversations,
        }

    dataset = dataset.map(add_conversations)
    print(dataset)
    dataset.save_to_disk(args.output)

if __name__ == "__main__":
    main()