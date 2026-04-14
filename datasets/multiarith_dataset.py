import json
import pandas as pd
import numpy as np
from typing import Union, List, Literal, Any, Dict


def load_multiarith(path: str = "datasets/MultiArith/MultiArith.json"):
    with open(path) as f:
        data = json.load(f)
    return data


def multiarith_data_process(dataset: list) -> list:
    """Process MultiArith dataset to task dict format."""
    list_data_dict = []
    for item in dataset:
        question = item["sQuestion"].strip()
        # lSolutions is a list, take the first
        answer = str(item["lSolutions"][0]) if item["lSolutions"] else ""
        list_data_dict.append({
            "task": question,
            "answer": answer,
            "equation": item["lEquations"][0] if item["lEquations"] else "",
            "index": item["iIndex"],
        })
    return list_data_dict


def multiarith_get_predict(pred_str: str) -> str:
    """Extract final numeric answer from model prediction."""
    if not pred_str:
        return ""
    # Try to find "The answer is ..." pattern
    if "the answer is" in pred_str.lower():
        parts = pred_str.lower().split("the answer is")
        candidate = parts[-1].strip().rstrip(".")
        # Extract number from end
        import re
        nums = re.findall(r'-?\d+\.?\d*', candidate)
        if nums:
            return nums[-1]
    # Try to find boxed
    if "boxed" in pred_str:
        import re
        ans = pred_str.split("boxed")[-1].strip()
        nums = re.findall(r'-?\d+\.?\d*', ans)
        if nums:
            return nums[-1]
    # Fallback: extract last number
    import re
    nums = re.findall(r'-?\d+\.?\d*', pred_str)
    if nums:
        return nums[-1]
    return ""


def _strip_string(s: str) -> str:
    """Normalize string for comparison."""
    if not s:
        return ""
    s = s.replace("\n", " ").replace(",", "").strip()
    # remove trailing dots
    while s and s[-1] in ".:-":
        s = s[:-1]
    return s.strip()
