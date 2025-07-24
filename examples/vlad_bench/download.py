"""Download VLADBench dataset."""

import argparse
from pathlib import Path
import json
from datasets import load_dataset, Dataset
import yaml

COUNTRIES = {
    "Amercia": "America",
    "America": "America",
    "Canada": "Canada",
    "China": "China",
    "Germany": "Germany",
    "Japan": "Japan",
    "LingoQA": None,
    "SODA": "China",
}


def process_subset(input_path: Path, subset_name: str):
    """Load a single subset from the input dataset."""
    task_path = input_path / subset_name

    items_path = Path(f"{task_path}_E.json")
    if not items_path.exists():
        print(f"Missing subset: {items_path}")
        return []
    items = json.load(items_path.open())

    metadata_path = task_path / "metadata.jsonl"
    metadata_f = metadata_path.open("w")
    for item in items:
        country = item["country"]
        assert country in COUNTRIES, (subset_name, item["id"], country)
        country = COUNTRIES[item["country"]]

        questions: list[str] = item["questions"]
        answers: list[str] = item["reference"]
        assert len(questions) == len(answers)
        for question, answer in zip(questions, answers):
            metadata = {
                "id": item["id"],
            }

            image_name, _, question = question.partition(";")
            if image_name.startswith("["):
                # Video not supported yet
                continue
            else:
                image_path = task_path / image_name
                assert image_path.exists(), (subset_name, item["id"], image_path)
                metadata["image"] = str(image_path)

            if country is not None:
                question = f"The image is from {country}. {question}"
            metadata["conversations"] = [
                {
                    "role": "system",
                    "content": "Answer the questions.",
                },
                {
                    "role": "user",
                    "content": json.dumps([
                        {
                            "type": "image",
                            "image": "image",
                        },
                        {
                            "type": "text",
                            "text": str(question),
                        },
                    ]),
                },
                {
                    "role": "assistant",
                    "content": str(answer),
                },
            ]

            json.dump(metadata, metadata_f)
            metadata_f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input directory.",
    )
    args = parser.parse_args()
    input_path: Path = args.input

    # Process tasks
    all_subsets: dict[str, dict[str, list[str]]] = json.load(
        (input_path / "all_tasks.json").open()
    )
    configs = []
    for k1, v1 in sorted(all_subsets.items()):
        for k2, v2 in sorted(v1.items()):
            for k3 in sorted(v2):
                subset_name = "/".join([k1, k2, k3])
                process_subset(input_path, subset_name)
                configs.append(
                    {
                        "config_name": "_".join(
                            [k.replace("_", "") for k in [k1, k2, k3]]
                        ),
                        "data_files": f"{subset_name}/metadata.jsonl",
                    }
                )

    # Write configs
    with (input_path / "README.md").open("w") as f:
        f.write("---\n")
        yaml.dump(
            {
                "configs": configs,
            },
            f,
        )
        f.write("---\n")


if __name__ == "__main__":
    main()
