import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import GDesigner.agents  # noqa: F401
import GDesigner.llm  # noqa: F401
import GDesigner.prompt  # noqa: F401
from GDesigner.graph.graph import Graph
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time
from datasets.MMLU.download import download
from datasets.mmlu_dataset import MMLUDataset
from experiments.evaluate_mmlu import evaluate
from experiments.usage_metrics import (
    reset_usage_metrics,
    usage_delta,
    usage_snapshot,
    write_run_metrics,
    zero_usage,
)


BASELINE_MODES = [
    "Vanilla",
    "CoT",
    "ComplexCoT",
    "SelfConsistencyCoT",
    "SelfConsistencyComplexCoT",
    "PHP",
    "Chain",
    "Star",
    "Tree",
    "FullConnected",
    "CompleteGraph",
    "Random",
    "AutoGen",
    "LLMBlender",
    "LLMDebate",
    "DyLAN",
    "GPTSwarm",
    "MetaGPT",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run one MMLU baseline method.")
    parser.add_argument("--mode", type=str, required=True, choices=BASELINE_MODES)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument("--domain", type=str, default="mmlu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--agent_nums", type=int, default=None)
    parser.add_argument("--sc_samples", type=int, default=10)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--limit_questions", type=int, default=153)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose topology and agent prompt/response logging.")
    return parser.parse_args()


def generate_tree_graph(num_nodes: int) -> List[List[int]]:
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for child in range(1, num_nodes):
        parent = (child - 1) // 2
        adj_matrix[parent][child] = 1
    return adj_matrix


def generate_layered_graph(num_nodes: int, layer_num: int = 2) -> List[List[int]]:
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    base_size = num_nodes // layer_num
    remainder = num_nodes % layer_num
    layers = []
    for layer in range(layer_num):
        size = base_size + (1 if layer < remainder else 0)
        layers.extend([layer] * size)
    random.shuffle(layers)
    for source in range(num_nodes):
        for target in range(num_nodes):
            if layers[target] == layers[source] + 1:
                adj_matrix[source][target] = 1
    return adj_matrix


def generate_star_graph(num_nodes: int) -> List[List[int]]:
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for target in range(1, num_nodes):
        adj_matrix[0][target] = 1
    return adj_matrix


def topology_kwargs(topology_mode: str, num_nodes: int) -> Dict[str, Any]:
    if topology_mode == "Independent":
        fixed_spatial_masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        fixed_temporal_masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(num_nodes)] for j in range(num_nodes)]
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(num_nodes)] for j in range(num_nodes)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(num_nodes)] for j in range(num_nodes)]
        fixed_temporal_masks = [[1 if i == 0 and j == num_nodes - 1 else 0 for i in range(num_nodes)] for j in range(num_nodes)]
    elif topology_mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(num_nodes)
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "Star":
        fixed_spatial_masks = generate_star_graph(num_nodes)
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    elif topology_mode == "Tree":
        fixed_spatial_masks = generate_tree_graph(num_nodes)
        fixed_temporal_masks = [[1 for _ in range(num_nodes)] for _ in range(num_nodes)]
    else:
        raise ValueError(f"Unsupported topology mode: {topology_mode}")

    return {
        "initial_spatial_probability": 0.5,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": 0.5,
        "fixed_temporal_masks": fixed_temporal_masks,
    }


def resolve_method_config(args) -> Dict[str, Any]:
    mode = args.mode
    config: Dict[str, Any] = {
        "mode": mode,
        "domain": args.domain,
        "num_rounds": args.num_rounds,
        "implementation_type": "native",
        "notes": "",
    }

    single_prompt_styles = {
        "Vanilla": "vanilla",
        "CoT": "cot",
        "ComplexCoT": "complex_cot",
        "PHP": "php",
    }
    if mode in single_prompt_styles:
        config.update({
            "agent_name": "BaselineAgent",
            "agent_count": args.agent_nums or 1,
            "decision_method": "FinalDirect",
            "topology_mode": "Independent",
            "prompt_style": single_prompt_styles[mode],
            "temperature": 0.0 if args.temperature is None else args.temperature,
            "implementation_type": "native" if mode != "PHP" else "adapted",
        })
        return config

    if mode == "SelfConsistencyCoT":
        config.update({
            "agent_name": "BaselineAgent",
            "agent_count": args.agent_nums or args.sc_samples,
            "decision_method": "FinalMajorVote",
            "topology_mode": "Independent",
            "prompt_style": "cot",
            "temperature": 1.0 if args.temperature is None else args.temperature,
        })
        return config

    if mode == "SelfConsistencyComplexCoT":
        config.update({
            "agent_name": "BaselineAgent",
            "agent_count": args.agent_nums or args.sc_samples,
            "decision_method": "FinalMajorVote",
            "topology_mode": "Independent",
            "prompt_style": "complex_cot",
            "temperature": 1.0 if args.temperature is None else args.temperature,
        })
        return config

    topology_by_mode = {
        "Chain": ("Chain", "native", ""),
        "Star": ("Star", "native", ""),
        "Tree": ("Tree", "native", ""),
        "FullConnected": ("FullConnected", "adapted", "DAG execution may reduce cyclic edges."),
        "CompleteGraph": ("FullConnected", "adapted", "DAG execution may reduce cyclic edges."),
        "Random": ("Random", "adapted", "DAG execution may reduce cyclic edges."),
        "AutoGen": ("Star", "adapted", "Uses the current graph coordinator-style configuration."),
        "LLMBlender": ("Independent", "adapted", "Uses independent answers with a final aggregation node."),
        "LLMDebate": ("Debate", "adapted", "Uses the current temporal debate-style configuration."),
        "DyLAN": ("Layered", "adapted", "Uses the current layered graph configuration."),
        "GPTSwarm": ("FullConnected", "adapted", "Uses the current full-connected graph configuration."),
    }
    if mode in topology_by_mode:
        topology_mode, implementation_type, notes = topology_by_mode[mode]
        config.update({
            "agent_name": "AnalyzeAgent",
            "agent_count": args.agent_nums or 5,
            "decision_method": "FinalRefer",
            "topology_mode": topology_mode,
            "prompt_style": None,
            "temperature": 1.0 if args.temperature is None else args.temperature,
            "implementation_type": implementation_type,
            "notes": notes,
        })
        return config

    if mode == "MetaGPT":
        config.update({
            "agent_name": None,
            "agent_count": args.agent_nums or 5,
            "decision_method": None,
            "topology_mode": None,
            "prompt_style": None,
            "temperature": args.temperature,
            "implementation_type": "skipped",
            "notes": "Not evaluated on this benchmark by default.",
        })
        return config

    raise ValueError(f"Unsupported mode: {mode}")


def build_graph(config: Dict[str, Any], args) -> Graph:
    agent_names = [config["agent_name"] for _ in range(config["agent_count"])]
    node_kwargs = [{} for _ in agent_names]
    if config["agent_name"] == "BaselineAgent":
        node_kwargs = [
            {
                "prompt_style": config["prompt_style"],
                "role": config["mode"],
            }
            for _ in agent_names
        ]

    kwargs = topology_kwargs(config["topology_mode"], len(agent_names))
    kwargs["node_kwargs"] = node_kwargs

    return Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=config["decision_method"],
        optimized_spatial=False,
        optimized_temporal=False,
        train_limit=0,
        sample_times=0,
        llm_temperature=config["temperature"],
        verbose=not args.quiet,
        **kwargs,
    )


async def main():
    args = parse_args()
    random.seed(args.seed)
    config = resolve_method_config(args)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(args.result_dir) if args.result_dir else Path(GDesigner_ROOT / "result" / "mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    safe_mode = args.mode.replace("/", "_")
    result_file = result_dir / f"{args.llm_name}_{current_time}_{safe_mode}.json"

    if config["implementation_type"] == "skipped":
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump({
                "Method": args.mode,
                "Status": "skipped",
                "Method_Config": config,
            }, file, indent=2, ensure_ascii=False)
        print(f"Skipped {args.mode}: {config['notes']}")
        print(f"Result file: {result_file}")
        return

    download()
    dataset_val = MMLUDataset("val")
    graph = build_graph(config, args)

    reset_usage_metrics()
    inference_start_usage = usage_snapshot()
    inference_start_ts = time.time()
    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
        result_file=result_file,
        method_name=args.mode,
        method_config=config,
    )
    inference_seconds = time.time() - inference_start_ts
    inference_usage = usage_delta(inference_start_usage)
    metrics_file = write_run_metrics(
        result_file,
        method_name=args.mode,
        method_config=config,
        llm_name=args.llm_name,
        dataset_name=dataset_val.__class__.__name__,
        split=dataset_val.split,
        score=score,
        training_usage=zero_usage(),
        training_seconds=0.0,
        inference_usage=inference_usage,
        inference_seconds=inference_seconds,
    )
    print(f"Score: {score}")
    print(f"Result file: {result_file}")
    print(f"Metrics file: {metrics_file}")


if __name__ == "__main__":
    asyncio.run(main())
