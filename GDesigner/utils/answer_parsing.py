import re
from typing import Any, Iterable


def extract_choice_answer(answer: Any, choices: Iterable[str] = ("A", "B", "C", "D")) -> str:
    choice_set = {choice.upper() for choice in choices}

    if isinstance(answer, list):
        for item in answer:
            parsed = extract_choice_answer(item, choice_set)
            if parsed:
                return parsed
        return ""

    if not isinstance(answer, str):
        raise Exception("Expected string")

    text = answer.strip()
    if not text:
        return ""

    choices_pattern = "".join(sorted(choice_set))
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    parsed = _parse_choice_from_leading_text(first_line, choices_pattern)
    if parsed:
        return parsed

    explicit_patterns = [
        rf"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:|：)\s*(?:\*\*)?\s*(?:option\s*)?[\(\[]?([{choices_pattern}])\b",
        rf"(?:final\s+answer|correct\s+answer|answer).{{0,80}}\boption\s+([{choices_pattern}])\b",
        rf"\boption\s+([{choices_pattern}])\b.{{0,80}}\b(?:correct|best|right)\b",
        rf"\b(?:choose|select|pick)\s*(?:option\s*)?([{choices_pattern}])\b",
    ]
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    for line in text.splitlines():
        parsed = _parse_choice_from_leading_text(line.strip(), choices_pattern)
        if parsed:
            return parsed

    return ""


def _parse_choice_from_leading_text(text: str, choices_pattern: str) -> str:
    normalized = text.strip().strip("`").strip("*").strip()
    if not normalized:
        return ""

    patterns = [
        rf"^(?:answer|final\s+answer|correct\s+answer)?\s*(?:is|:|：)?\s*(?:option\s*)?[\(\[]?([{choices_pattern}])[\)\]]?\s*(?:$|[\.。,:：\-)])",
        rf"^(?:\*\*)?\s*(?:option\s*)?([{choices_pattern}])(?:\*\*)?\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""
