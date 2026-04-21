#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# MultiArith has one 600-example JSON file. These commands use a deterministic
# 40-example optimization split and evaluate the remaining 560 examples.

python experiments/run_multiarith.py --mode Chain --agent_nums 5 --batch_size 4 --num_iterations 10 --num_rounds 3 --llm_name gpt-4o --optimized_spatial --train_limit 40 --sample_times 10 --limit_questions 560 --tau 1e-2 --zeta 1e-1 --eval_edge_threshold 0.5 --temperature 1.0 --quiet

python experiments/run_multiarith_baseline.py --mode Vanilla --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 0.0 --quiet
python experiments/run_multiarith_baseline.py --mode CoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 0.0 --quiet
python experiments/run_multiarith_baseline.py --mode ComplexCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 0.0 --quiet
python experiments/run_multiarith_baseline.py --mode PHP --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 0.0 --quiet
python experiments/run_multiarith_baseline.py --mode SelfConsistencyCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --sc_samples 10 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode SelfConsistencyComplexCoT --batch_size 4 --num_rounds 1 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --sc_samples 10 --temperature 1.0 --quiet

python experiments/run_multiarith_baseline.py --mode Chain --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode Star --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode Tree --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode CompleteGraph --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode Random --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet

python experiments/run_multiarith_baseline.py --mode AutoGen --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode LLMBlender --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode LLMDebate --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode DyLAN --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
python experiments/run_multiarith_baseline.py --mode GPTSwarm --agent_nums 5 --batch_size 4 --num_rounds 3 --llm_name gpt-4o --train_limit 40 --limit_questions 560 --temperature 1.0 --quiet
