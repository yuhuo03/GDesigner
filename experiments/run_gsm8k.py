import sys
import os
import argparse
import time
import asyncio
from pathlib import Path
from typing import Union, Literal, List
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
from datasets.gsm8k_dataset import GSM8KDataset
from experiments.train_gsm8k import train
from experiments.evaluate_gsm8k import evaluate
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on GSM8K")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                                 'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--llm_name', type=str, default="gpt-4o",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="gsm8k",
                        help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--limit_questions', type=int, default=None,
                        help="Limit number of questions to evaluate. Default None (all).")
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help="Evaluation batch size. Default 4.")
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args


def get_kwargs(mode, N):
    initial_spatial_probability = 0.5
    fixed_spatial_masks = None
    initial_temporal_probability = 0.5
    fixed_temporal_masks = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0] * N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i + 1, N):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Math Expert'}]
    elif mode == 'FullConnected' or mode == 'FakeFullConnected' or mode == 'FakeAGFull':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
        node_kwargs = None
    elif mode == 'Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
        node_kwargs = None
    elif mode == 'Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Mesh' or mode == 'FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Star' or mode == 'FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None

    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs}


async def main():
    args = parse_args()

    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode, len(agent_names))

    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(GDesigner_ROOT / "result" / "gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    dataset_train = GSM8KDataset('train')
    dataset_val = GSM8KDataset('val')

    if args.optimized_spatial or args.optimized_temporal:
        await train(graph=graph, dataset=dataset_train, num_iters=args.num_iterations, num_rounds=args.num_rounds,
                    lr=args.lr, batch_size=args.batch_size)

    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
        num_rounds=args.num_rounds,
        limit_questions=args.limit_questions,
        eval_batch_size=args.eval_batch_size,
        result_file=result_file,
    )
    print(f"Score: {score}")
    print(f"Result file: {result_file}")


if __name__ == '__main__':
    asyncio.run(main())
