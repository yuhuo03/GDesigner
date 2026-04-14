import json, re
from typing import Dict, Any, List


def load_aqua(path: str = "datasets/AQuA/AQuA.jsonl") -> list:
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def aqua_data_process(dataset: list) -> list:
    """Process AQuA dataset to task dict format."""
    list_data_dict = []
    for item in dataset:
        question = item["question"].strip()
        options = item["options"]
        correct = item["correct"].strip()
        list_data_dict.append({
            "task": question,
            "options": options,
            "correct": correct,
            "rationale": item.get("rationale", ""),
        })
    return list_data_dict


def aqua_record_to_input(record: Dict) -> Dict[str, Any]:
    """Format AQuA question into prompt input."""
    question = record["task"]
    options = record["options"]
    options_text = "\n".join(options)
    demo_question = f"{question}\n{options_text}"
    return {"task": demo_question}


def aqua_postprocess_answer(answer: str, options: list = None) -> str:
    """Extract option letter (A-E) from model response.

    Strategy:
    1. "answer is A/B/C/D/E" pattern (standalone letter) → return it.
    2. Formula matching (most important for AQuA): if options provided,
       normalize LaTeX vs Unicode and find which option's formula
       appears in the response.
    3. Standalone single letter A-E at end of response → return it.
    """
    if not answer:
        return ""
    answer = answer.strip()
    lower = answer.lower()

    # 1. "answer is A/B/C/D/E" standalone pattern
    if "answer is" in lower:
        idx = lower.find("answer is")
        candidate = answer[idx + len("answer is"):].strip().rstrip(".").strip()
        if candidate and candidate[0] in "ABCDE" and len(candidate) <= 3:
            return candidate[0]

    # 2. Formula matching against options (handles LaTeX vs Unicode)
    if options:
        # Normalize answer: collapse whitespace, strip trailing punctuation
        answer_norm = re.sub(r'\s+', ' ', answer).strip()
        for opt in options:
            m = re.match(r'^([A-E])\)(.+)', opt.strip())
            if not m:
                continue
            letter, formula = m.group(1), m.group(2).strip()
            # Try multiple normalizations of the formula
            for f in [
                formula,
                re.sub(r'\s+', '', formula),                              # no whitespace
                formula.lower().replace('\u221a', 'sqrt'),                # unicode -> word
                formula.replace('\u221a', r'\\sqrt\{').replace('}', '}'), # unicode -> latex
            ]:
                # Simple substring match (case-insensitive)
                if f.lower() in answer_norm.lower():
                    return letter
                # Regex with escaped special chars
                fp = re.escape(f)
                if re.search(fp, answer_norm, re.IGNORECASE):
                    return letter

    # 3. Standalone single A-E letter at end of response
    for c in reversed(answer):
        if c in "ABCDE":
            return c
        if c.isalnum():
            break

    return ""


def aqua_record_to_target_answer(record: Dict) -> str:
    """Return the correct answer letter."""
    return record["correct"]
