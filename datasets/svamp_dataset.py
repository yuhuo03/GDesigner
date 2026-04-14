import json
from typing import Dict, Any


def load_svamp(path: str = "datasets/SVAMP/SVAMP.json") -> list:
    with open(path) as f:
        data = json.load(f)
    return data


def svamp_data_process(dataset: list) -> list:
    """Process SVAMP dataset to task dict format."""
    list_data_dict = []
    for item in dataset:
        # Combine Body and Question
        body = item["Body"].strip()
        question = item["Question"].strip()
        full_question = f"{body} {question}".strip()
        answer = str(item["Answer"]) if item["Answer"] not in [None, ""] else ""
        list_data_dict.append({
            "task": full_question,
            "answer": answer,
            "equation": item.get("Equation", ""),
            "type": item.get("Type", ""),
            "id": item.get("ID", ""),
        })
    return list_data_dict


def svamp_get_predict(pred_str: str) -> str:
    """Extract final numeric answer from model prediction."""
    if not pred_str:
        return ""
    import re
    # Try "the answer is" pattern first
    lower = pred_str.lower()
    if "the answer is" in lower:
        parts = lower.split("the answer is")
        candidate = parts[-1].strip().rstrip(".")
        nums = re.findall(r'-?\d+\.?\d*', candidate)
        if nums:
            return nums[-1]
    # Try boxed
    if "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1].strip()
        nums = re.findall(r'-?\d+\.?\d*', ans)
        if nums:
            return nums[-1]
    # Fallback: last number
    nums = re.findall(r'-?\d+\.?\d*', pred_str)
    if nums:
        return nums[-1]
    return ""


def _strip_string(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace(",", "").strip()
    while s and s[-1] in ".:-":
        s = s[:-1]
    return s.strip()
