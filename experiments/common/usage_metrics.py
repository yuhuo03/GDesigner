import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union

from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens


def reset_usage_metrics() -> None:
    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()


def usage_snapshot() -> Dict[str, Union[float, int]]:
    prompt_tokens = int(PromptTokens.instance().value)
    completion_tokens = int(CompletionTokens.instance().value)
    return {
        "Cost": float(Cost.instance().value),
        "PromptTokens": prompt_tokens,
        "CompletionTokens": completion_tokens,
        "TotalTokens": prompt_tokens + completion_tokens,
    }


def zero_usage() -> Dict[str, Union[float, int]]:
    return {
        "Cost": 0.0,
        "PromptTokens": 0,
        "CompletionTokens": 0,
        "TotalTokens": 0,
    }


def usage_delta(
    start: Mapping[str, Union[float, int]],
    end: Optional[Mapping[str, Union[float, int]]] = None,
) -> Dict[str, Union[float, int]]:
    end = usage_snapshot() if end is None else end
    return {
        "Cost": float(end["Cost"]) - float(start["Cost"]),
        "PromptTokens": int(end["PromptTokens"]) - int(start["PromptTokens"]),
        "CompletionTokens": int(end["CompletionTokens"]) - int(start["CompletionTokens"]),
        "TotalTokens": int(end["TotalTokens"]) - int(start["TotalTokens"]),
    }


def combine_usage(usages: Iterable[Mapping[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
    combined = zero_usage()
    for usage in usages:
        combined["Cost"] = float(combined["Cost"]) + float(usage.get("Cost", 0.0))
        combined["PromptTokens"] = int(combined["PromptTokens"]) + int(usage.get("PromptTokens", 0))
        combined["CompletionTokens"] = int(combined["CompletionTokens"]) + int(usage.get("CompletionTokens", 0))
        combined["TotalTokens"] = int(combined["TotalTokens"]) + int(usage.get("TotalTokens", 0))
    return combined


def usage_with_time(
    usage: Mapping[str, Union[float, int]],
    wall_clock_seconds: float,
) -> Dict[str, Union[float, int]]:
    return {
        "Cost": float(usage.get("Cost", 0.0)),
        "PromptTokens": int(usage.get("PromptTokens", 0)),
        "CompletionTokens": int(usage.get("CompletionTokens", 0)),
        "TotalTokens": int(usage.get("TotalTokens", 0)),
        "WallClockSeconds": float(wall_clock_seconds),
        "WallClockMinutes": float(wall_clock_seconds) / 60.0,
    }


def result_stats(
    result_file: Union[str, Path],
    valid_predictions: Optional[Iterable[str]] = None,
    track_invalid_predictions: bool = True,
) -> Dict[str, Union[float, int]]:
    valid_prediction_set = set(valid_predictions or {"A", "B", "C", "D"})
    result_path = Path(result_file)
    if not result_path.exists():
        return {
            "Total_Solved": 0,
            "Total_Executed": 0,
            "Accuracy": 0.0,
            "Invalid_Predictions": 0,
        }

    with open(result_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        return {
            "Total_Solved": 0,
            "Total_Executed": 0,
            "Accuracy": 0.0,
            "Invalid_Predictions": 0,
        }

    total_executed = len(records)
    total_solved = sum(1 for record in records if record.get("Solved"))
    invalid_predictions = 0
    if track_invalid_predictions:
        invalid_predictions = sum(
            1
            for record in records
            if record.get("Pred_Answer") not in valid_prediction_set
        )
    accuracy = total_solved / total_executed if total_executed else 0.0
    return {
        "Total_Solved": total_solved,
        "Total_Executed": total_executed,
        "Accuracy": accuracy,
        "Invalid_Predictions": invalid_predictions,
    }


def write_run_metrics(
    result_file: Union[str, Path],
    *,
    method_name: str,
    method_config: Optional[Dict[str, Any]] = None,
    llm_name: Optional[str] = None,
    dataset_name: str = "MMLU",
    split: str = "val",
    score: Optional[float] = None,
    training_usage: Optional[Mapping[str, Union[float, int]]] = None,
    training_seconds: float = 0.0,
    inference_usage: Optional[Mapping[str, Union[float, int]]] = None,
    inference_seconds: float = 0.0,
    csv_file: Optional[Union[str, Path]] = None,
    valid_predictions: Optional[Iterable[str]] = None,
    track_invalid_predictions: bool = True,
) -> Path:
    result_path = Path(result_file)
    metrics_path = result_path.with_name(f"{result_path.stem}_metrics.json")
    stats = result_stats(
        result_path,
        valid_predictions=valid_predictions,
        track_invalid_predictions=track_invalid_predictions,
    )

    training = usage_with_time(training_usage or zero_usage(), training_seconds)
    inference = usage_with_time(inference_usage or zero_usage(), inference_seconds)
    overall_usage = combine_usage([training, inference])
    overall = usage_with_time(overall_usage, training_seconds + inference_seconds)
    final_score = stats["Accuracy"] if score is None else score

    metrics = {
        "Method": method_name,
        "Method_Config": method_config or {},
        "Dataset": dataset_name,
        "Split": split,
        "LLM": llm_name,
        "Result_File": str(result_path),
        "Score": final_score,
        "Result_Stats": stats,
        "Metrics": {
            "Training": training,
            "Inference": inference,
            "Overall": overall,
        },
        "Table_Fields": {
            "Accuracy (%)": float(final_score) * 100.0,
            "Training Prompt Tokens": training["PromptTokens"],
            "Inference Prompt Tokens": inference["PromptTokens"],
            "Overall Prompt Tokens": overall["PromptTokens"],
            "Training Total Tokens": training["TotalTokens"],
            "Inference Total Tokens": inference["TotalTokens"],
            "Overall Total Tokens": overall["TotalTokens"],
            "Training Time (min)": training["WallClockMinutes"],
            "Inference Time (min)": inference["WallClockMinutes"],
            "Overall Time (min)": overall["WallClockMinutes"],
            "Cost (USD)": overall["Cost"],
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    csv_path = Path(csv_file) if csv_file is not None else result_path.parent / "run_metrics.csv"
    append_run_metrics_csv(csv_path, metrics_path, metrics)
    return metrics_path


def append_run_metrics_csv(
    csv_file: Union[str, Path],
    metrics_file: Union[str, Path],
    metrics: Mapping[str, Any],
) -> None:
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    stats = metrics["Result_Stats"]
    training = metrics["Metrics"]["Training"]
    inference = metrics["Metrics"]["Inference"]
    overall = metrics["Metrics"]["Overall"]

    row = {
        "Method": metrics["Method"],
        "LLM": metrics["LLM"],
        "Dataset": metrics["Dataset"],
        "Split": metrics["Split"],
        "Score": metrics["Score"],
        "Accuracy": stats["Accuracy"],
        "Total_Solved": stats["Total_Solved"],
        "Total_Executed": stats["Total_Executed"],
        "Invalid_Predictions": stats["Invalid_Predictions"],
        "Training_Cost": training["Cost"],
        "Training_PromptTokens": training["PromptTokens"],
        "Training_CompletionTokens": training["CompletionTokens"],
        "Training_TotalTokens": training["TotalTokens"],
        "Training_WallClockSeconds": training["WallClockSeconds"],
        "Inference_Cost": inference["Cost"],
        "Inference_PromptTokens": inference["PromptTokens"],
        "Inference_CompletionTokens": inference["CompletionTokens"],
        "Inference_TotalTokens": inference["TotalTokens"],
        "Inference_WallClockSeconds": inference["WallClockSeconds"],
        "Overall_Cost": overall["Cost"],
        "Overall_PromptTokens": overall["PromptTokens"],
        "Overall_CompletionTokens": overall["CompletionTokens"],
        "Overall_TotalTokens": overall["TotalTokens"],
        "Overall_WallClockSeconds": overall["WallClockSeconds"],
        "Result_File": metrics["Result_File"],
        "Metrics_File": str(metrics_file),
    }

    fieldnames = list(row.keys())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
