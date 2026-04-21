import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import List, Literal, Union

import GDesigner.agents  # noqa: F401
import GDesigner.llm  # noqa: F401
import GDesigner.prompt  # noqa: F401
from GDesigner.graph.graph import Graph
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time
from datasets.multiarith_dataset import MULTIARITH_DEFAULT_PATH, MultiArithDataset
from experiments.multiarith.evaluate import evaluate
from experiments.multiarith.train import train
from experiments.common.usage_metrics import (
    reset_usage_metrics,
    usage_delta,
    usage_snapshot,
    write_run_metrics,
    zero_usage,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on MultiArith")
    parser.add_argument("--data_path", "--dataset_json", dest="data_path", type=str, default=MULTIARITH_DEFAULT_PATH)
    parser.add_argument("--mode", type=str, default="Chain",
                        choices=["DirectAnswer", "FullConnected", "CompleteGraph", "Random", "Chain", "Debate",
                                 "Layered", "Star", "Tree", "Mesh", "FakeFullConnected", "FakeRandom",
                                 "FakeChain", "FakeStar", "FakeMesh", "FakeAGRandom", "FakeAGFull"],
                        help="Mode of operation. Default is 'Chain'.")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--agent_names", nargs="+", type=str, default=["MathSolver"],
                        help="Specify agent names as a list of strings")
    parser.add_argument("--agent_nums", nargs="+", type=int, default=[5],
                        help="Specify the number of agents for each name in agent_names")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument("--imp_per_iterations", type=int, default=5,
                        help="Kept for CLI compatibility with other experiment runners.")
    parser.add_argument("--num_rounds", type=int, default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument("--pruning_rate", type=float, default=0.25,
                        help="Kept for CLI compatibility with other experiment runners.")
    parser.add_argument("--llm_name", type=str, default="gpt-4o",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument("--domain", type=str, default="multiarith",
                        help="Domain (the same as dataset name), default 'multiarith'")
    parser.add_argument("--decision_method", type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument("--optimized_spatial", action="store_true")
    parser.add_argument("--optimized_temporal", action="store_true")
    parser.add_argument("--train_limit", type=int, default=40,
                        help="Number of MultiArith examples used for topology optimization. Default 40.")
    parser.add_argument("--sample_times", type=int, default=10,
                        help="Topology samples per training query. Default 10.")
    parser.add_argument("--tau", type=float, default=1e-2,
                        help="Sampling temperature for learned spatial topology.")
    parser.add_argument("--zeta", type=float, default=1e-1,
                        help="Weight for the low-rank sparsity regularizer.")
    parser.add_argument("--limit_questions", type=int, default=None,
                        help="Limit number of evaluation questions. Default None evaluates the full selected split.")
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test", "heldout", "full"],
                        help="MultiArith evaluation split. val and heldout alias test.")
    parser.add_argument("--eval_edge_threshold", type=float, default=0.5,
                        help="Deterministic edge threshold used during evaluation.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="LLM temperature for MultiArith multi-agent runs. Default 1.0.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm for topology optimization. Default 1.0.")
    parser.add_argument("--seed", type=int, default=888,
                        help="Seed used for deterministic MultiArith split.")
    parser.add_argument("--quiet", action="store_true",
                        help="Disable verbose topology and agent prompt/response logging.")
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result" / "multiarith"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    if args.optimized_temporal and not args.optimized_spatial:
        print("MultiArith topology optimization learns spatial edges; enabling --optimized_spatial because --optimized_temporal was set.")
        args.optimized_spatial = True

    random.seed(args.seed)
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode, len(agent_names))

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        tau=args.tau,
        zeta=args.zeta,
        train_limit=args.train_limit,
        sample_times=args.sample_times,
        eval_edge_threshold=args.eval_edge_threshold,
        llm_temperature=args.temperature,
        verbose=not args.quiet,
        **kwargs,
    )

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(GDesigner_ROOT / "result" / "multiarith")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    dataset_train = None
    if args.optimized_spatial or args.optimized_temporal:
        dataset_train = MultiArithDataset("train", data_path=args.data_path, seed=args.seed, train_limit=args.train_limit)
    dataset_eval = MultiArithDataset(args.eval_split, data_path=args.data_path, seed=args.seed, train_limit=args.train_limit)

    reset_usage_metrics()
    training_usage = zero_usage()
    training_seconds = 0.0

    if args.optimized_spatial or args.optimized_temporal:
        training_start_usage = usage_snapshot()
        training_start_ts = time.time()
        await train(
            graph=graph,
            dataset=dataset_train,
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size,
            train_limit=args.train_limit,
            sample_times=args.sample_times,
            grad_clip=args.grad_clip,
        )
        training_seconds = time.time() - training_start_ts
        training_usage = usage_delta(training_start_usage)

    inference_start_usage = usage_snapshot()
    inference_start_ts = time.time()
    score = await evaluate(
        graph=graph,
        dataset=dataset_eval,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.batch_size,
        result_file=result_file,
        method_name="GDesigner",
        method_config=vars(args),
    )
    inference_seconds = time.time() - inference_start_ts
    inference_usage = usage_delta(inference_start_usage)
    metrics_file = write_run_metrics(
        result_file,
        method_name="GDesigner",
        method_config=vars(args),
        llm_name=args.llm_name,
        dataset_name=dataset_eval.__class__.__name__,
        split=dataset_eval.split,
        score=score,
        training_usage=training_usage,
        training_seconds=training_seconds,
        inference_usage=inference_usage,
        inference_seconds=inference_seconds,
        track_invalid_predictions=False,
    )
    print(f"Score: {score}")
    print(f"Result file: {result_file}")
    print(f"Metrics file: {metrics_file}")


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["CompleteGraph"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Tree"],
        Literal["Mesh"],
        Literal["FakeFullConnected"],
        Literal["FakeRandom"],
        Literal["FakeChain"],
        Literal["FakeStar"],
        Literal["FakeMesh"],
        Literal["FakeAGRandom"],
        Literal["FakeAGFull"],
    ],
    N: int,
):
    initial_spatial_probability = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

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

    def generate_mesh_graph(num_nodes: int) -> List[List[int]]:
        adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for source in range(num_nodes):
            for target in range(source + 1, num_nodes):
                adj_matrix[source][target] = 1
        return adj_matrix

    def generate_star_graph(num_nodes: int) -> List[List[int]]:
        adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for target in range(1, num_nodes):
            adj_matrix[0][target] = 1
        return adj_matrix

    def generate_tree_graph(num_nodes: int) -> List[List[int]]:
        adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]
        for child in range(1, num_nodes):
            parent = (child - 1) // 2
            adj_matrix[parent][child] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[0 for _ in range(N)] for _ in range(N)]
        node_kwargs = [{"role": "Mathematical Analyst"} for _ in range(N)]
    elif mode in ("FullConnected", "CompleteGraph", "FakeFullConnected", "FakeAGFull"):
        fixed_spatial_masks = [[1 if target != source else 0 for target in range(N)] for source in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode in ("Random", "FakeRandom", "FakeAGRandom"):
        fixed_spatial_masks = [[random.randint(0, 1) if target != source else 0 for target in range(N)] for source in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode in ("Chain", "FakeChain"):
        fixed_spatial_masks = [[1 if target == source + 1 else 0 for target in range(N)] for source in range(N)]
        fixed_temporal_masks = [[1 if target == 0 and source == N - 1 else 0 for target in range(N)] for source in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode in ("Mesh", "FakeMesh"):
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode in ("Star", "FakeStar"):
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Tree":
        fixed_spatial_masks = generate_tree_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    if "Fake" in mode and "AG" not in mode:
        node_kwargs = [{"role": "Fake"} if i % 2 == N % 2 else {"role": "Mathematical Analyst"} for i in range(N)]
    elif "Fake" in mode and "AG" in mode:
        node_kwargs = [{"role": "Fake"} if i % 2 == N % 2 else {"role": None} for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
