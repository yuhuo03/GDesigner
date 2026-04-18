import torch
import time
import asyncio
from typing import Iterator, List
import copy
import numpy as np

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens


async def train(graph: Graph,
                dataset,
                num_iters: int = 10,
                num_rounds: int = 1,
                lr: float = 0.1,
                batch_size: int = 4,
                ) -> None:

    def infinite_data_loader():
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                record = dataset[idx]
                yield record

    loader = infinite_data_loader()
    optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=lr)
    graph.gcn.train()

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * '-')
        start_ts = time.time()
        answer_log_probs = []
        correct_answers = []

        for _ in range(batch_size):
            record = next(loader)
            realized_graph = copy.deepcopy(graph)
            realized_graph.gcn = graph.gcn
            realized_graph.mlp = graph.mlp
            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
            correct_answers.append(dataset.record_to_target_answer(record))

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            is_solved = float(answer) == float(correct_answer)
            utilities.append(is_solved)
            single_loss = -log_prob * is_solved
            loss_list.append(single_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"answers: {answers}")
        print(f"correct_answers: {correct_answers}")
        print(f"utilities: {utilities}")
        print(f"loss: {total_loss.item():.4f}")
        print(f"Batch time {time.time() - start_ts:.3f}s")
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
