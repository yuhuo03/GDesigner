import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np


GSM8K_DEFAULT_DATA_DIR = "datasets/gsm8k"
_NUMBER_PATTERN = r"-?(?:\d+\.\d+|\d+|\.\d+)(?:/\d+)?"


class GSM8KDataset:
    def __init__(
        self,
        split: Union[Literal["train"], Literal["val"], Literal["test"], Literal["full"]],
        data_dir: Union[str, Path] = GSM8K_DEFAULT_DATA_DIR,
        seed: int = 42,
    ) -> None:
        self._split = "val" if split == "test" else split
        self.data_dir = Path(data_dir)
        self.seed = seed
        self._data: List[Dict[str, Any]] = self._load_data(self._resolve_data_file())

    @staticmethod
    def get_domain() -> str:
        return "gsm8k"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._data[int(index)]
        if isinstance(index, slice):
            return self._data[index]
        raise TypeError(f"indices must be int or slice, not {type(index)}")

    def _resolve_data_file(self) -> Path:
        split_files = {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "full": "gsm8k.jsonl",
        }
        if self._split not in split_files:
            raise ValueError(f"Unsupported GSM8K split: {self._split}")
        data_file = self.data_dir / split_files[self._split]
        if not data_file.exists():
            raise FileNotFoundError(f"Missing GSM8K {self._split} split at {data_file}.")
        return data_file

    def _load_data(self, data_path: Path) -> List[Dict[str, Any]]:
        with open(data_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file if line.strip()]
        print(f"[GSM8KDataset] Loaded {len(data)} samples from {data_path}")
        return data

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": record["question"]}

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        return gsm8k_postprocess_answer(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        raw_answer = record["answer"]
        parsed_answer = gsm8k_postprocess_answer(raw_answer)
        if parsed_answer:
            return parsed_answer
        raw_answer_list = raw_answer.split("####")
        return normalize_numeric_answer(raw_answer_list[-1])


def gsm8k_postprocess_answer(answer: Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        for item in answer:
            parsed = gsm8k_postprocess_answer(item)
            if parsed:
                return parsed
        return ""

    if not isinstance(answer, str):
        return ""

    pred_str = answer.strip()
    if not pred_str:
        return ""

    candidates = []
    answer_patterns = [
        r"(?:final answer|the answer is|answer is|answer)\s*[:：]?\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, pred_str, flags=re.IGNORECASE)
        candidates.extend(matches)

    boxed_value = _extract_boxed_value(pred_str)
    if boxed_value:
        candidates.append(boxed_value)

    for candidate in reversed(candidates):
        numeric = _extract_answer_candidate_number(candidate)
        if numeric:
            return normalize_numeric_answer(numeric)
    return normalize_numeric_answer(_extract_last_number(pred_str))


def normalize_numeric_answer(answer: Any) -> str:
    text = str(answer).strip()
    if not text:
        return ""
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("\\", "").replace("{", "").replace("}", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(".,;:()[]")
    if text.startswith("+"):
        text = text[1:]
    return text


def gsm8k_answer_equal(predicted: Union[str, List[str]], target: Union[str, List[str]]) -> bool:
    pred = normalize_numeric_answer(_parsed_or_first_raw(predicted))
    gold = normalize_numeric_answer(_parsed_or_first_raw(target))
    if not pred or not gold:
        return pred == gold

    pred_number = _to_number(pred)
    gold_number = _to_number(gold)
    if pred_number is not None and gold_number is not None:
        return pred_number == gold_number
    return pred == gold


def gsm_get_predict(pred_str):
    return gsm8k_postprocess_answer(pred_str)


def _parsed_or_first_raw(answer: Union[str, List[str]]) -> str:
    parsed = gsm8k_postprocess_answer(answer)
    if parsed:
        return parsed
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return answer


def _extract_last_number(text: str) -> str:
    normalized = text.replace(",", "")
    matches = re.findall(_NUMBER_PATTERN, normalized)
    return matches[-1] if matches else ""


def _extract_answer_candidate_number(text: str) -> str:
    normalized = text.replace(",", "").strip()
    direct_number = re.match(rf"^({_NUMBER_PATTERN})(?:\s*(?:[.!。]|$))", normalized)
    if direct_number:
        return direct_number.group(1)
    return _extract_last_number(normalized)


def _extract_boxed_value(text: str) -> str:
    marker = "boxed"
    lower = text.lower()
    if marker not in lower:
        return ""
    start = lower.rfind(marker) + len(marker)
    rest = text[start:].lstrip()
    if not rest:
        return ""
    if rest[0] != "{":
        return rest.split("$")[0].strip()

    depth = 1
    value = []
    for char in rest[1:]:
        if char == "{":
            depth += 1
            value.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
            value.append(char)
        else:
            value.append(char)
    return "".join(value).strip()


def _to_number(text: str):
    text = normalize_numeric_answer(text)
    if not text:
        return None
    try:
        if "/" in text and text.count("/") == 1:
            return Fraction(text)
        return Decimal(text)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
