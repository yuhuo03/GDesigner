# GDesigner

## Overview

GDesigner provides topology optimization for multi-agent LLM collaboration. Core implementation code is in `GDesigner`, dataset adapters are in `datasets`, and experiment entry points are in `experiments`.

## Quick Start

### Install packages

```bash
conda create -n gdesigner python=3.10
conda activate gdesigner
pip install -r requirements.txt
```

### Add API keys in `template.env` and change its name to `.env`

```python
BASE_URL = "" # the BASE_URL of OpenAI LLM backend
API_KEY = "" # for OpenAI LLM backend
```

### Prepare Datasets

The supported datasets are MMLU, AQuA, GSM8K, MultiArith, SVAMP and HumanEval. MMLU and AQuA include download helpers:

```bash
python datasets/MMLU/download.py
python datasets/AQuA/download.py
```

Other dataset files are expected under these paths:

```bash
datasets/gsm8k/train.jsonl
datasets/gsm8k/val.jsonl
datasets/MultiArith/MultiArith.json
datasets/SVAMP/SVAMP.json
datasets/humaneval/humaneval-py.jsonl
```

### Run Experiment Scripts

The shell scripts run the configured GDesigner command first and then the dataset's baseline commands. They assume you are already in the `gdesigner` conda environment and running from the repository root.

```bash
bash experiments/run_mmlu_experiments.sh
bash experiments/run_aqua_experiments.sh
bash experiments/run_gsm8k_experiments.sh
bash experiments/run_multiarith_experiments.sh
bash experiments/run_svamp_experiments.sh
bash experiments/run_humaneval_experiments.sh
```

### Run GDesigner Directly

The current GDesigner commands use a single communication round (`--num_rounds 1`), optimize spatial topology only, sample 40 optimization examples, use 10 topology samples per query, and evaluate with a deterministic edge threshold of `0.5`.

```bash
python experiments/run_mmlu.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 153 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
python experiments/run_aqua.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 254 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
python experiments/run_gsm8k.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 1279 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
python experiments/run_multiarith.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 560 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
python experiments/run_svamp.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 960 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
python experiments/run_humaneval.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 1 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet
```

### Run Baselines

Each dataset has a separate baseline runner named `experiments/run_<dataset>_baseline.py`. Use `--mode` to select a baseline, for example:

```bash
python experiments/run_mmlu_baseline.py --mode Vanilla --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 153 --temperature 0.0 --quiet
python experiments/run_gsm8k_baseline.py --mode CoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 1279 --temperature 0.0 --quiet
```

Results and usage metrics are written under `result/<dataset>/`.

## Acknowledgement

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm).
