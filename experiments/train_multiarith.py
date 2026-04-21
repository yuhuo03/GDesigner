import asyncio
import copy
import time
from typing import Iterator, List

import numpy as np
import torch

from GDesigner.graph.graph import Graph
from GDesigner.utils.globals import CompletionTokens, Cost, PromptTokens
from datasets.multiarith_dataset import multiarith_answer_equal


async def train(
    graph: Graph,
    dataset,
    num_iters: int = 10,
    num_rounds: int = 3,
    lr: float = 0.1,
    batch_size: int = 4,
    train_limit: int = 40,
    sample_times: int = 10,
    grad_clip: float = 1.0,
) -> None:
    def infinite_data_loader() -> Iterator:
        data_size = min(len(dataset), train_limit) if train_limit is not None else len(dataset)
        perm = np.random.permutation(data_size)
        while True:
            for idx in perm:
                yield dataset[idx]

    trainable_params = graph.topology_parameters()
    if not trainable_params:
        raise ValueError("No trainable topology parameters were selected.")

    loader = infinite_data_loader()
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    graph.set_edge_sampling(False)
    graph.set_topology_train(True)

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        start_ts = time.time()
        answer_log_probs = []
        correct_answers = []
        realized_graphs = []

        for _, record in zip(range(batch_size), loader):
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            correct_answer = dataset.record_to_target_answer(record)
            for _ in range(sample_times):
                realized_graph = copy.deepcopy(graph)
                realized_graph.share_parameters_from(graph)
                realized_graphs.append(realized_graph)
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
                correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        regularization_losses: List[float] = []

        for raw_answer, log_prob, correct_answer, realized_graph in zip(
            raw_answers,
            log_probs,
            correct_answers,
            realized_graphs,
        ):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            utility = float(multiarith_answer_equal(answer, correct_answer))
            utilities.append(utility)
            regularization_loss = getattr(realized_graph, "topology_regularization_loss", torch.tensor(0.0))
            regularization_losses.append(float(regularization_loss.detach().cpu()))
            single_loss = -log_prob * utility + regularization_loss
            loss_list.append(single_loss)
            print(f"correct answer:{correct_answer}")

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        if not torch.isfinite(total_loss):
            print(f"Skipping optimizer step because loss is not finite: {total_loss.item()}")
            continue
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip, error_if_nonfinite=False)
        if not torch.isfinite(grad_norm):
            print(f"Skipping optimizer step because gradient norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            continue
        optimizer.step()

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("regularization_losses:", regularization_losses)
        print("loss:", total_loss.item())
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
