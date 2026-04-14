import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import copy
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Union, Literal
import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from datasets.aqua_dataset import load_aqua, aqua_data_process, aqua_record_to_input, aqua_postprocess_answer, aqua_record_to_target_answer


def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def dataloader(data_list, batch_size, i_batch):
    start = i_batch * batch_size
    end = start + batch_size
    if start >= len(data_list):
        return None
    return data_list[start:end]


def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on AQuA")
    parser.add_argument("--dataset_json", type=str, default="datasets/AQuA/AQuA.jsonl")
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                                 'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5])
    parser.add_argument('--num_iterations', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--pruning_rate', type=float, default=0.25)
    parser.add_argument('--domain', type=str, default="aqua")
    parser.add_argument('--decision_method', type=str, default="FinalRefer")
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')
    parser.add_argument('--limit_questions', type=int, default=None)
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result" / "aqua"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


def get_kwargs(mode, N):
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
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
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh':
        fixed_spatial_masks = [[1 if i > j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Programming Expert'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


async def main():
    args = parse_args()

    raw_data = load_aqua(args.dataset_json)
    dataset = aqua_data_process(raw_data)
    if args.limit_questions:
        dataset = dataset[:args.limit_questions]
    print(f"[AQuA] Total questions: {len(dataset)}")

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(GDesigner_ROOT / "result" / "aqua")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=args.decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    graph.gcn.train()
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr)

    total_solved, total_executed = 0, 0
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for i_batch in range(num_batches):
        print(f"\nBatch {i_batch}/{num_batches}", 80 * '-')
        start_ts = time.time()

        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if current_batch is None:
            break

        answer_log_probs = []
        batch_records = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            batch_records.append((realized_graph, record))
            input_dict = aqua_record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, args.num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)

        loss_list = []
        utilities = []
        data = load_result(result_file)

        for idx, (realized_graph, record) in enumerate(batch_records):
            raw_answers, log_prob = raw_results[idx]
            answer = raw_answers[0] if isinstance(raw_answers, list) else raw_answers
            predict_answer = aqua_postprocess_answer(answer, options=record["options"])
            true_answer = aqua_record_to_target_answer(record)

            is_solved = (predict_answer == true_answer)
            total_solved += float(is_solved)
            total_executed += 1
            accuracy = total_solved / total_executed
            utility = float(is_solved)
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)

            updated_item = {
                "Question": record["task"],
                "Options": record["options"],
                "GT_Answer": true_answer,
                "Pred_Answer": predict_answer,
                "Rationale": record.get("rationale", ""),
                "Solved": bool(is_solved),
                "Total_Solved": total_solved,
                "Total_Executed": total_executed,
                "Accuracy": accuracy,
            }
            data.append(updated_item)
            print(f"  [{idx}] GT={true_answer} Pred={predict_answer} {'✓' if is_solved else '✗'} Acc={accuracy:.3f}")

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        total_loss = torch.mean(torch.stack(loss_list))
        if args.optimized_spatial or args.optimized_temporal:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"  Loss: {total_loss.item():.4f}")

        if (i_batch + 1) % args.num_iterations == 0 and (args.optimized_spatial or args.optimized_temporal):
            args.optimized_spatial = False
            args.optimized_temporal = False
            total_solved = 0
            total_executed = 0
            graph.gcn.eval()
            print("  → Switched to eval mode")

        print(f"  Time: {time.time() - start_ts:.1f}s")
        print(f"  Cost: ${Cost.instance().value:.4f} | PromptTokens: {PromptTokens.instance().value} | CompletionTokens: {CompletionTokens.instance().value}")


if __name__ == "__main__":
    asyncio.run(main())
