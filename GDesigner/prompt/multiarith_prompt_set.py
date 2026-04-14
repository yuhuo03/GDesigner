"""
PromptSet for MultiArith dataset.
Multi-step equation-based arithmetic reasoning.
"""
from typing import Dict, Any, Union, List
import itertools

from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.common import get_combine_materials


roles = itertools.cycle([
    'Mathematical Analyst',
    'Math Solver',
    'Programming Expert',
    'Inspector',
])


# Role → description (canonical keys)
ROLE_DESCRIPTION = {
    "Mathematical Analyst": (
        "You are a mathematical analyst skilled at arithmetic, algebraic reasoning, and equation solving. "
        "You will be given a math problem, analysis and code from other agents. "
        "Analyze step by step, substitute values, and report the final result. "
        "The last line of your output must be: The answer is <number> (no units)."
    ),
    "Math Solver": (
        "You are a math expert. You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output must be: The answer is <number> (no units)."
    ),
    "Programming Expert": (
        "You are a programming expert skilled at Python. "
        "Write a function with no arguments that returns/prints the answer to the math problem. "
        "Output only a Python code block."
    ),
    "Inspector": (
        "You are an Inspector. Check logic, calculations, and code correctness. "
        "Report the final answer step by step. "
        "The last line of your output must be: The answer is <number> (no units)."
    ),
    "Fake": (
        "You are a liar who only tells lies. No matter what others say, give a wrong answer."
    ),
}


# Multi-agent communication topology (who informs whom)
ROLE_CONNECTION = [
    ('Mathematical Analyst', 'Math Solver'),
    ('Mathematical Analyst', 'Programming Expert'),
    ('Mathematical Analyst', 'Inspector'),
    ('Math Solver',          'Programming Expert'),
    ('Math Solver',          'Mathematical Analyst'),
    ('Programming Expert',    'Inspector'),
    ('Programming Expert',    'Math Solver'),
    ('Inspector',             'Mathematical Analyst'),
    ('Inspector',             'Math Solver'),
    ('Math Solver',           'Inspector'),
]


# Few-shot examples (sampled from MultiArith dataset)
FEW_SHOT_DATA = {
    "Mathematical Analyst": (
        'Q: There are 15 trees. Workers will plant 3 trees in each row. How many rows?\n'
        'A:\n'
        'Step 1: 15 trees total, 3 per row.\n'
        'Step 2: 15 / 3 = 5.\n'
        'The answer is 5\n\n'
        'Q: Leah had 32 chocolates and her sister had 42. They ate 35. How many left?\n'
        'A:\n'
        'Step 1: 32 + 42 = 74 total.\n'
        'Step 2: 74 - 35 = 39.\n'
        'The answer is 39'
    ),
    "Math Solver": (
        'Q: There are 15 trees. Workers will plant 3 trees in each row. How many rows?\n'
        'A:\n'
        'Hint: The answer is near to 5.\n'
        'Step 1: 15 / 3 = 5.\n'
        'The answer is 5\n\n'
        'Q: Leah had 32 chocolates and her sister had 42. They ate 35. How many left?\n'
        'A:\n'
        'Hint: The answer is near to 39.\n'
        'Step 1: 32 + 42 = 74.\n'
        'Step 2: 74 - 35 = 39.\n'
        'The answer is 39'
    ),
    "Programming Expert": (
        'Q: There are 15 trees. Workers will plant 3 trees in each row. How many rows?\n'
        'A:\n'
        '```python\n'
        'def solve():\n'
        '    return 15 // 3\n'
        '\n'
        'print(solve())\n'
        '```\n\n'
        'Q: Leah had 32 chocolates and her sister had 42. They ate 35. How many left?\n'
        'A:\n'
        '```python\n'
        'def solve():\n'
        '    return 32 + 42 - 35\n'
        '\n'
        'print(solve())\n'
        '```'
    ),
    "Inspector": (
        'Q: There are 15 trees. Workers will plant 3 trees in each row. How many rows?\n'
        'A:\n'
        'Check: 15 / 3 = 5 rows. Correct.\n'
        'The answer is 5\n\n'
        'Q: Leah had 32 chocolates and her sister had 42. They ate 35. How many left?\n'
        'A:\n'
        'Check: 32+42=74, 74-35=39. Correct.\n'
        'The answer is 39'
    ),
}


@PromptSetRegistry.register('multiarith')
class MultiArithPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role=None):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Mathematical Analyst"])

    def get_description(self, role=None):
        return self.get_constraint(role)

    def get_role_connection(self):
        return ROLE_CONNECTION

    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        shots = FEW_SHOT_DATA.get(role, FEW_SHOT_DATA["Mathematical Analyst"])
        return "{}\n\n---END OF EXAMPLES---\n\nQ:{}".format(shots, question)

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math problem and solutions from other agents. "
            "Find the most reliable answer. "
            "The last line of your output must be: The answer is <number> (no units)."
        )

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker for math problems."

    @staticmethod
    def get_decision_few_shot():
        return ""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return "Q: {}\nSolution: {}\nFeedback: {}\nRewrite the solution.".format(
            question, solution, feedback)

    @staticmethod
    def get_query_prompt(question):
        return "Target question:\n{}".format(question)

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return "Extract from file:\n{}\n\nFile:\n{}".format(query, file)

    @staticmethod
    def get_websearch_prompt(question, query):
        return "Web search for question:\n{}\nTarget:\n{}".format(question, query)

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return (
"# Adversarial Challenge Task\n\n"
f"## Target Question:\n---\n{question}\n---\n\n"
"## 📋 Instructions:\n"
"1. Identify potential weaknesses or edge cases in the question.\n"
"2. Analyze the question from an adversarial perspective.\n"
"3. Provide an incorrect or misleading solution.\n"
"4. The last line of your output must be: The answer is <wrong_number> (no units).\n"
"Do not reveal that you are giving a wrong answer."
        )

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return "Q: {}\nQuery: {}\nResults: {}\nSummarize.".format(question, query, results)

    @staticmethod
    def get_reflect_prompt(question, answer):
        return "Q: {}\nAnswer: {}\nReflect.".format(question, answer)

    @staticmethod
    def get_combine_materials(materials):
        return get_combine_materials(materials)

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            return ""
        answer = answer.strip()
        if not answer:
            return ""
        if "answer is" in answer.lower():
            parts = answer.lower().split("answer is")
            if len(parts) > 1:
                token = parts[-1].strip().strip(": ").split()[0] if parts[-1].strip() else ""
                return token
        lines = [l.strip() for l in answer.split("\n") if l.strip()]
        return lines[-1] if lines else answer
