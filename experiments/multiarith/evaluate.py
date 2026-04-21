import asyncio
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from tqdm import tqdm

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens
from datasets.multiarith_dataset import multiarith_answer_equal
from experiments.common.accuracy import Accuracy


def load_result(result_file: Path) -> List[Dict[str, Any]]:
    if not result_file.exists():
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump([], file)

    with open(result_file, "r", encoding="utf-8") as file:
        return json.load(file)


async def evaluate(
    graph: Graph,
    dataset,
    num_rounds: int = 3,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 4,
    result_file: Optional[Union[str, Path]] = None,
    method_name: Optional[str] = None,
    method_config: Optional[Dict[str, Any]] = None,
) -> float:
    method_label = method_name or "gdesigner"
    print(f"Evaluating {method_label} on {dataset.__class__.__name__} split {dataset.split}")

    graph.set_topology_train(False)
    graph.set_edge_sampling(True)
    accuracy = Accuracy()
    total_correct = 0
    total_executed = 0
    result_path = Path(result_file) if result_file is not None else None
    result_data = load_result(result_path) if result_path is not None else None

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size)) if eval_batch_size else 0

    for _, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80 * "-")

        start_ts = time.time()
        answer_log_probs = []
        realized_graphs = []

        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.share_parameters_from(graph)
            realized_graphs.append(realized_graph)
            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers = [result[0] for result in raw_results]
        execution_traces = [getattr(realized_graph, "execution_trace", None) for realized_graph in realized_graphs]
        print(f"Batch time {time.time() - start_ts:.3f}")

        for raw_answer, record, execution_trace in zip(raw_answers, record_batch, execution_traces):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            is_correct = multiarith_answer_equal(answer, correct_answer)
            total_correct += int(is_correct)
            total_executed += 1
            accuracy.update("1" if is_correct else "0", "1")
            accuracy.print()
            if result_data is not None:
                result_data.append({
                    **({"Method": method_name} if method_name is not None else {}),
                    **({"Method_Config": method_config} if method_config is not None else {}),
                    "Question": record["task"],
                    "Index": record.get("index", ""),
                    "Equation": record.get("equation", ""),
                    "GT_Answer": correct_answer,
                    "Pred_Answer": answer,
                    "Raw_Answer": raw_answer,
                    "Solved": bool(is_correct),
                    "Total_Solved": total_correct,
                    "Total_Executed": total_executed,
                    "Accuracy": total_correct / total_executed,
                    "Execution_Trace": execution_trace,
                })

        if result_path is not None:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, indent=2, ensure_ascii=False)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

    if total_executed == 0:
        print("No examples evaluated.")
        print("Done!")
        return 0.0

    accuracy.print()
    print("Done!")
    return accuracy.get()
