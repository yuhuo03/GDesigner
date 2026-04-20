import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np

from GDesigner.utils.answer_parsing import extract_choice_answer


AQUA_CHOICES = ("A", "B", "C", "D", "E")


class AQuADataset:
    def __init__(
        self,
        split: Literal["train", "dev", "test", "val"],
        data_dir: Union[str, Path] = "datasets/AQuA",
        seed: int = 888,
    ) -> None:
        self._split = "test" if split == "val" else split
        self.data_dir = Path(data_dir)
        self.seed = seed
        self._records = self._load_records()

    @staticmethod
    def get_domain() -> str:
        return "aqua"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._records[int(index)]
        if isinstance(index, slice):
            return self._records[index]
        raise TypeError(f"indices must be int or slice, not {type(index)}")

    def _load_records(self) -> List[Dict[str, Any]]:
        data_file = self._resolve_data_file()
        records = aqua_data_process(load_aqua(data_file))
        if self._split == "train":
            rng = np.random.default_rng(self.seed)
            indices = rng.permutation(len(records))
            records = [records[int(index)] for index in indices]
        return records

    def _resolve_data_file(self) -> Path:
        split_files = {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json",
        }
        if self._split not in split_files:
            raise ValueError(f"Unsupported AQuA split: {self._split}")

        data_file = self.data_dir / split_files[self._split]
        if data_file.exists():
            return data_file

        raise FileNotFoundError(
            f"Missing AQuA {self._split} split at {data_file}. "
            "Run `python datasets/AQuA/download.py` to download the full AQuA-RAT splits."
        )

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        question = record["question"]
        options_text = "\n".join(record["options"])
        return {"task": f"{question}\n{options_text}"}

    def postprocess_answer(
        self,
        answer: Union[str, List[str]],
        options: List[str] | None = None,
    ) -> str:
        return aqua_postprocess_answer(answer, options=options)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        correct_answer = record["correct"]
        assert correct_answer in AQUA_CHOICES, (
            f"A-E expected but got {correct_answer} "
            f"of type {type(correct_answer)} record={record}"
        )
        return correct_answer


def load_aqua(path: Union[str, Path] = "datasets/AQuA/test.json") -> list:
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def aqua_data_process(dataset: list) -> list:
    list_data_dict = []
    for item in dataset:
        question = item["question"].strip()
        options = [str(option).strip() for option in item["options"]]
        correct = item["correct"].strip()
        list_data_dict.append({
            "question": question,
            "task": question,
            "options": options,
            "A": _option_text(options, "A"),
            "B": _option_text(options, "B"),
            "C": _option_text(options, "C"),
            "D": _option_text(options, "D"),
            "E": _option_text(options, "E"),
            "correct": correct,
            "rationale": item.get("rationale", ""),
        })
    return list_data_dict


def _option_text(options: List[str], letter: str) -> str:
    prefix = f"{letter})"
    for option in options:
        if option.strip().startswith(prefix):
            return option.strip()[len(prefix):].strip()
    return ""


def aqua_record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
    return AQuADataset.record_to_input(record)


def aqua_postprocess_answer(
    answer: Union[str, List[str]],
    options: List[str] | None = None,
) -> str:
    if isinstance(answer, list):
        for item in answer:
            parsed = aqua_postprocess_answer(item, options=options)
            if parsed:
                return parsed
        return ""

    if not isinstance(answer, str):
        return ""

    parsed = extract_choice_answer(answer, choices=AQUA_CHOICES)
    if parsed:
        return parsed

    if options:
        matched = _match_answer_to_options(answer, options)
        if matched:
            return matched

    return ""


def _match_answer_to_options(answer: str, options: List[str]) -> str:
    answer_norm = _normalize_option_text(answer)
    if not answer_norm:
        return ""

    for option in options:
        match = re.match(r"^([A-E])\)(.+)", option.strip())
        if not match:
            continue
        letter, value = match.group(1), match.group(2)
        value_norm = _normalize_option_text(value)
        if value_norm and (value_norm in answer_norm or answer_norm in value_norm):
            return letter
    return ""


def _normalize_option_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("\\", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("√", "sqrt").replace("−", "-").replace("–", "-")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.strip(".,;:()[]")
    return normalized


def aqua_record_to_target_answer(record: Dict[str, Any]) -> str:
    return AQuADataset.record_to_target_answer(record)
