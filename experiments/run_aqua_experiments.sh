#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# AQuA-RAT uses the official train/test split. GDesigner samples 40
# optimization queries from train.json and evaluates all 254 test examples.

python experiments/run_aqua.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 3 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 254 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet

python experiments/run_aqua_baseline.py --mode Vanilla --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --temperature 0.0 --quiet
python experiments/run_aqua_baseline.py --mode CoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --temperature 0.0 --quiet
python experiments/run_aqua_baseline.py --mode ComplexCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --temperature 0.0 --quiet
python experiments/run_aqua_baseline.py --mode PHP --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --temperature 0.0 --quiet
python experiments/run_aqua_baseline.py --mode SelfConsistencyCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --sc_samples 10 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode SelfConsistencyComplexCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --limit_questions 254 --sc_samples 10 --temperature 1.0 --quiet

python experiments/run_aqua_baseline.py --mode Chain --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode Star --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode Tree --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode CompleteGraph --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode Random --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet

python experiments/run_aqua_baseline.py --mode AutoGen --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode LLMBlender --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode LLMDebate --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode DyLAN --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
python experiments/run_aqua_baseline.py --mode GPTSwarm --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --limit_questions 254 --temperature 1.0 --quiet
