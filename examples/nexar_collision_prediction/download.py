"""Download Nexar collision prediction dataset."""

import argparse
import json
from pathlib import Path
import datasets
from rich import print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(
        "nexar-ai/nexar_collision_prediction", split="train"
    )
    print(dataset)
    dataset = dataset.cast_column("video", datasets.Video(decode=False))

    def process_sample(sample: dict) -> dict:
        video_path = sample["video"]["path"]
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

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
            "video": video_path,
        }

    dataset = list(map(process_sample, dataset))
    dataset = datasets.Dataset.from_generator(lambda: dataset)
    print(dataset)
    dataset.save_to_disk(str(output_dir))


if __name__ == "__main__":
    main()
