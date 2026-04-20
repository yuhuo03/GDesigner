import os
from pathlib import Path
from typing import Iterable

import requests


SPLIT_FILES = {
    "train": "train.json",
    "dev": "dev.json",
    "test": "test.json",
}

BASE_URL = "https://raw.githubusercontent.com/google-deepmind/AQuA/master"


def download(
    data_dir: str | os.PathLike | None = None,
    required_splits: Iterable[str] = ("train", "dev", "test"),
) -> None:
    target_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parent
    target_dir.mkdir(parents=True, exist_ok=True)

    for split in required_splits:
        if split not in SPLIT_FILES:
            raise ValueError(f"Unsupported AQuA split: {split}")

        target_file = target_dir / SPLIT_FILES[split]
        if target_file.exists():
            continue

        url = f"{BASE_URL}/{SPLIT_FILES[split]}"
        print(f"Downloading {url}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(target_file, "wb") as file:
            file.write(response.content)
        print(f"Saved to {target_file}")


if __name__ == "__main__":
    download()
