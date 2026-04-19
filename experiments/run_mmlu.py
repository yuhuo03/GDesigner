import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import argparse
import random
import time
from pathlib import Path
from typing import Union, Literal, List

from GDesigner.graph.graph import Graph
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.train_mmlu import train
from experiments.evaluate_mmlu import evaluate
from experiments.usage_metrics import (
    reset_usage_metrics,
    usage_delta,
    usage_snapshot,
    write_run_metrics,
    zero_usage,
)
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time



def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='Chain',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'Chain'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=3,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--llm_name', type=str, default="gpt-4o",
                        help="Model name, None runs the default ChatGPT4")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--train_limit', type=int, default=40,
                        help="Number of MMLU dev examples used for topology optimization. Default 40.")
    parser.add_argument('--sample_times', type=int, default=10,
                        help="Topology samples per training query. Default 10.")
    parser.add_argument('--tau', type=float, default=1e-2,
                        help="Sampling temperature for learned spatial topology.")
    parser.add_argument('--zeta', type=float, default=1e-1,
                        help="Weight for the low-rank sparsity regularizer.")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="Limit number of validation questions. Default 153 matches the configured MMLU test count.")
    parser.add_argument('--eval_edge_threshold', type=float, default=0.5,
                        help="Deterministic edge threshold used during evaluation.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help="LLM temperature for MMLU multi-agent runs. Default 1.0.")
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help="Max gradient norm for topology optimization. Default 1.0.")
    parser.add_argument('--quiet', action='store_true',
                        help="Disable verbose topology and agent prompt/response logging.")
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args

async def main():
    args = parse_args()
    if args.optimized_temporal and not args.optimized_spatial:
        print("MMLU topology optimization learns spatial edges; enabling --optimized_spatial because --optimized_temporal was set.")
        args.optimized_spatial = True
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    
    graph = Graph(domain=args.domain,
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
                  **kwargs)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(GDesigner_ROOT / "result" / "mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    download()
    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')

    reset_usage_metrics()
    training_usage = zero_usage()
    training_seconds = 0.0
    
    if args.optimized_spatial or args.optimized_temporal:
        training_start_usage = usage_snapshot()
        training_start_ts = time.time()
        await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
                    lr=args.lr,batch_size=args.batch_size,train_limit=args.train_limit,
                    sample_times=args.sample_times,grad_clip=args.grad_clip)
        training_seconds = time.time() - training_start_ts
        training_usage = usage_delta(training_start_usage)
        
    inference_start_usage = usage_snapshot()
    inference_start_ts = time.time()
    score = await evaluate(
        graph=graph,
        dataset=dataset_val,
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
        dataset_name=dataset_val.__class__.__name__,
        split=dataset_val.split,
        score=score,
        training_usage=training_usage,
        training_seconds=training_seconds,
        inference_usage=inference_usage,
        inference_seconds=inference_seconds,
    )
    print(f"Score: {score}")
    print(f"Result file: {result_file}")
    print(f"Metrics file: {metrics_file}")



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
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
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
        node_kwargs = None
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
        node_kwargs = None
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
        node_kwargs = None

    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == "__main__":
    asyncio.run(main())
