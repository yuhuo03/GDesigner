import re
from typing import Union, Literal, Any, Dict, List
import json
import numpy as np
from pathlib import Path


class GSM8KDataset:
    def __init__(self,
                 split: Union[Literal['train'], Literal['val']],
                 ) -> None:
        self._split = split
        data_path = Path(f"datasets/gsm8k/{split}.jsonl")
        self._data: List[Dict[str, Any]] = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'gsm8k'

    def _load_data(self, data_path: Path) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(42)
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        data = list(data)
        data = data  # keep original order for reproducibility
        print(f"[GSM8KDataset] Loaded {len(data)} samples from {data_path}")
        return data

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        return self._data[index]

    @staticmethod
    def record_to_input(record: Dict[str, Any]) -> Dict[str, Any]:
        return {"task": record["question"]}

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        pred_str = answer
        if 'The answer is ' in pred_str:
            pred = pred_str.split('The answer is ')[-1].strip()
        elif 'the answer is ' in pred_str:
            pred = pred_str.split('the answer is ')[-1].strip()
        elif 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if ans and ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
                pred = GSM8KDataset._strip_string(a)
            else:
                a = ans.split('$')[0].strip()
                pred = GSM8KDataset._strip_string(a)
        else:
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if len(pred) >= 1:
                pred = pred[-1]
            else:
                pred = ''

        if pred != "":
            if pred[-1] == ".":
                pred = pred[:-1]
            if pred[-1] == "/":
                pred = pred[:-1]

        pred = GSM8KDataset._strip_string(pred)

        if 'boxed' in pred:
            ans = pred.split('boxed')[-1]
            if ans and ans[0] == '{':
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{':
                        stack += 1
                        a += c
                    elif c == '}':
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
                pred = GSM8KDataset._strip_string(a)
            else:
                a = ans.split('$')[0].strip()
                pred = GSM8KDataset._strip_string(a)

        if pred.isdigit():
            return pred
        else:
            matches = re.findall(r'\d+', pred)
            return matches[-1] if matches else '0'

    @staticmethod
    def record_to_target_answer(record: Dict[str, Any]) -> str:
        raw_answer = record["answer"]
        raw_answer_list = raw_answer.split("\n####")
        return raw_answer_list[-1].replace(",", "").strip()

    @staticmethod
    def _strip_string(string: str) -> str:
        string = string.replace("\n", "")
        string = string.replace("\\!", "")
        string = string.replace("\\\\", "\\")
        string = string.replace("tfrac", "frac")
        string = string.replace("dfrac", "frac")
        string = string.replace("\\left", "")
        string = string.replace("\\right", "")
        string = string.replace("^{\\circ}", "")
        string = string.replace("^\\circ", "")
        string = string.replace("\\$", "")
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            if len(splits) == 2:
                string = splits[0]
        string = string.replace("\\%", "")
        string = string.replace("\%", "")
        if len(string.split("=")) == 2:
            parts = string.split("=")
            if len(parts[0]) <= 2:
                string = parts[1]
        if string.startswith("."):
            string = "0" + string
        if string == "0.5":
            string = "\\frac{1}{2}"
        string = string.replace(" ", "")
        return string


# Backwards-compatible re-export (used by MathSolver agent)
def gsm_get_predict(pred_str):
    """Alias for GSM8KDataset.postprocess_answer for backwards compatibility."""
    return GSM8KDataset.postprocess_answer(pred_str)
