import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np


MULTIARITH_DEFAULT_PATH = "datasets/MultiArith/MultiArith.json"
_NUMBER_PATTERN = r"-?(?:\d+\.\d+|\d+|\.\d+)"


class MultiArithDataset:
    def __init__(
        self,
        split: Union[
            Literal["train"],
            Literal["test"],
            Literal["val"],
            Literal["heldout"],
            Literal["full"],
        ],
        data_path: Union[str, Path] = MULTIARITH_DEFAULT_PATH,
        seed: int = 888,
        train_limit: int = 40,
    ) -> None:
        self._split = "test" if split in ("val", "heldout") else split
        self.data_path = Path(data_path)
        self.seed = seed
        self.train_limit = train_limit
        self._data = self._load_split()

    @staticmethod
    def get_domain() -> str:
        return "multiarith"

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

    def _load_split(self) -> List[Dict[str, Any]]:
        records = multiarith_data_process(load_multiarith(self.data_path))
        if self._split == "full":
            selected_records = records
        elif self._split in ("train", "test"):
            train_size = min(max(int(self.train_limit or 0), 0), len(records))
            rng = np.random.default_rng(self.seed)
            indices = [int(index) for index in rng.permutation(len(records))]
            if self._split == "train":
                selected_indices = indices[:train_size]
            else:
                selected_indices = indices[train_size:]
            selected_records = [records[index] for index in selected_indices]
        else:
            raise ValueError(f"Unsupported MultiArith split: {self._split}")

        print(f"[MultiArithDataset] Loaded {len(selected_records)} samples from {self.data_path} split {self._split}")
        return selected_records

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": record["task"]}

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        return multiarith_postprocess_answer(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        return normalize_numeric_answer(record["answer"])


def load_multiarith(path: Union[str, Path] = MULTIARITH_DEFAULT_PATH):
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing MultiArith data file at {data_path}.")
    with open(data_path, encoding="utf-8") as file:
        data = json.load(file)
    return data


def multiarith_data_process(dataset: list) -> List[Dict[str, Any]]:
    records = []
    for item in dataset:
        question = item["sQuestion"].strip()
        answer = str(item["lSolutions"][0]) if item.get("lSolutions") else ""
        records.append({
            "task": question,
            "answer": normalize_numeric_answer(answer),
            "equation": item["lEquations"][0] if item.get("lEquations") else "",
            "index": item["iIndex"],
        })
    return records


def multiarith_postprocess_answer(answer: Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        for item in answer:
            parsed = multiarith_postprocess_answer(item)
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
        r"(?:final answer|the answer is|answer is|answer)\s*[:]?\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
    ]
    for pattern in answer_patterns:
        candidates.extend(re.findall(pattern, pred_str, flags=re.IGNORECASE))

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
    number = _to_number(text)
    if number is not None:
        if number == number.to_integral_value():
            return str(int(number))
        return format(number.normalize(), "f").rstrip("0").rstrip(".")
    return text


def multiarith_answer_equal(predicted: Union[str, List[str]], target: Union[str, List[str]]) -> bool:
    pred = normalize_numeric_answer(_parsed_or_first_raw(predicted))
    gold = normalize_numeric_answer(_parsed_or_first_raw(target))
    if not pred or not gold:
        return pred == gold

    pred_number = _to_number(pred)
    gold_number = _to_number(gold)
    if pred_number is not None and gold_number is not None:
        return pred_number == gold_number
    return pred == gold


def multiarith_get_predict(pred_str: Union[str, List[str]]) -> str:
    return multiarith_postprocess_answer(pred_str)


def _parsed_or_first_raw(answer: Union[str, List[str]]) -> str:
    parsed = multiarith_postprocess_answer(answer)
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
    direct_number = re.match(rf"^({_NUMBER_PATTERN})(?:\s*(?:[.!]|$))", normalized)
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
    text = str(text).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None
