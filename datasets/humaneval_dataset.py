import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np


HUMANEVAL_DEFAULT_PATH = "datasets/humaneval/humaneval-py.jsonl"


class HumanEvalDataset:
    def __init__(
        self,
        split: Literal["train", "test", "val", "full", "heldout"],
        data_path: Union[str, Path] = HUMANEVAL_DEFAULT_PATH,
        seed: int = 888,
        train_limit: int = 40,
    ) -> None:
        self._split = "test" if split in ("val", "heldout") else split
        self.data_path = Path(data_path)
        self.seed = seed
        self.train_limit = train_limit
        self._records = self._load_records()

    @staticmethod
    def get_domain() -> str:
        return "humaneval"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self._records[int(index)]
        if isinstance(index, slice):
            return self._records[index]
        raise TypeError(f"indices must be int or slice, not {type(index)}")

    def _load_records(self) -> List[Dict[str, Any]]:
        records = load_humaneval(self.data_path)
        if self._split == "full":
            return records
        if self._split not in ("train", "test"):
            raise ValueError(f"Unsupported HumanEval split: {self._split}")

        train_size = min(max(int(self.train_limit or 0), 0), len(records))
        rng = np.random.default_rng(self.seed)
        indices = [int(index) for index in rng.permutation(len(records))]

        if self._split == "train":
            selected_indices = indices[:train_size]
        else:
            selected_indices = indices[train_size:]
        return [records[index] for index in selected_indices]

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": record["prompt"]}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        return humaneval_postprocess_answer(answer)

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        return record["prompt"] + record.get("canonical_solution", "")


def load_humaneval(path: Union[str, Path] = HUMANEVAL_DEFAULT_PATH) -> List[Dict[str, Any]]:
    data_file = Path(path)
    if not data_file.exists():
        raise FileNotFoundError(f"Missing HumanEval data file at {data_file}.")

    records = []
    with open(data_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def humaneval_record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
    return HumanEvalDataset.record_to_input(record)


def humaneval_record_to_target_answer(record: Dict[str, Any]) -> str:
    return HumanEvalDataset.record_to_target_answer(record)


def humaneval_postprocess_answer(answer: Union[str, List[str]]) -> str:
    return extract_python_code(answer)


def extract_python_code(answer: Union[str, List[str]]) -> str:
    if isinstance(answer, list):
        for item in reversed(answer):
            parsed = extract_python_code(item)
            if parsed:
                return parsed
        return ""

    if not isinstance(answer, str):
        return ""

    text = answer.strip()
    if not text:
        return ""

    code_blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if code_blocks:
        for block in reversed(code_blocks):
            code = _clean_code(block)
            if _looks_like_python_solution(code):
                return code
        return _clean_code(code_blocks[-1])

    return _clean_code(text)


def _clean_code(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python|py)?\s*", "", code, flags=re.IGNORECASE)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def _looks_like_python_solution(code: str) -> bool:
    return bool(re.search(r"^\s*(def|class)\s+", code, flags=re.MULTILINE))
