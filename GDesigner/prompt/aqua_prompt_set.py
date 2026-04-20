"""
PromptSet for AQuA (Allen AI Question Answering) dataset.
Multiple-choice math/logic reasoning with 5 options (A-E).
"""
from typing import Dict, Any, Union, List
import itertools

from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.common import get_combine_materials
from GDesigner.utils.answer_parsing import extract_choice_answer


roles = itertools.cycle([
    'Mathematical Analyst',
    'Math Solver',
    'Programming Expert',
    'Inspector',
])


ROLE_DESCRIPTION = {
    "Mathematical Analyst": (
        "You are a mathematical analyst skilled at arithmetic, algebraic reasoning, and logic. "
        "You will be given a multiple-choice math problem with options A-E and analysis from other agents. "
        "Analyze step by step, then output the letter of the best option from A, B, C, D, and E. "
        "The first line of your output must contain only one letter: A, B, C, D, or E."
    ),
    "Math Solver": (
        "You are a math expert. You will be given a multiple-choice math problem with options A-E and hints from other agents. "
        "Solve step by step, then output the letter of the best option from A, B, C, D, and E. "
        "The first line of your output must contain only one letter: A, B, C, D, or E."
    ),
    "Programming Expert": (
        "You are a programming expert skilled at Python. "
        "Given a multiple-choice math problem, write code to compute and verify each option, "
        "then output the letter of the correct choice from A, B, C, D, and E. "
        "The first line of your output must contain only one letter: A, B, C, D, or E."
    ),
    "Inspector": (
        "You are an Inspector. Verify logic and calculations, then output the letter of the correct option. "
        "The first line of your output must contain only one letter: A, B, C, D, or E."
    ),
    "Fake": (
        "You are a liar who only tells lies. No matter what others say, give a wrong option letter."
    ),
}


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


# Few-shot examples for AQuA (multiple choice)
FEW_SHOT_DATA = {
    "Mathematical Analyst": (
        'Q: If a bakery sells 240 loaves a day and each loaf is sold for $3, how much revenue in dollars do they earn in a week?\n'
        'Options: A) 5040  B) 720  C) 1000  D) 1440  E) 2000\n'
        'A: Let\'s solve step by step. Daily revenue = 240 × $3 = $720. Weekly revenue = 720 × 7 = $5040.\n'
        'The answer is A\n\n'
        'Q: A train travels 180 miles in 3 hours. What is its average speed in mph?\n'
        'Options: A) 36  B) 60  C) 90  D) 120  E) 540\n'
        'A: Speed = distance / time = 180 / 3 = 60 mph.\n'
        'The answer is B'
    ),
    "Math Solver": (
        'Q: If a bakery sells 240 loaves a day and each loaf is sold for $3, how much revenue in dollars do they earn in a week?\n'
        'Options: A) 5040  B) 720  C) 1000  D) 1440  E) 2000\n'
        'A: Hint: The answer is near to A.\n'
        '240 × 3 = 720 per day. 720 × 7 = 5040 per week.\n'
        'The answer is A\n\n'
        'Q: A train travels 180 miles in 3 hours. What is its average speed in mph?\n'
        'Options: A) 36  B) 60  C) 90  D) 120  E) 540\n'
        'A: Hint: The answer is near to B.\n'
        'Speed = 180 / 3 = 60 mph.\n'
        'The answer is B'
    ),
    "Programming Expert": (
        'Q: If a bakery sells 240 loaves a day and each loaf is sold for $3, how much revenue in dollars do they earn in a week?\n'
        'Options: A) 5040  B) 720  C) 1000  D) 1440  E) 2000\n'
        'A:\n'
        '```python\n'
        'revenue = 240 * 3 * 7\n'
        '# A) 5040  B) 720  C) 1000  D) 1440  E) 2000\n'
        'answer = "A"\n'
        '```\n\n'
        'Q: A train travels 180 miles in 3 hours. What is its average speed in mph?\n'
        'Options: A) 36  B) 60  C) 90  D) 120  E) 540\n'
        'A:\n'
        '```python\n'
        'speed = 180 / 3\n'
        '# A) 36  B) 60  C) 90  D) 120  E) 540\n'
        'answer = "B"\n'
        '```'
    ),
    "Inspector": (
        'Q: If a bakery sells 240 loaves a day and each loaf is sold for $3, how much revenue in dollars do they earn in a week?\n'
        'Options: A) 5040  B) 720  C) 1000  D) 1440  E) 2000\n'
        'A: Check: 240×3×7=5040. Correct option A.\n'
        'The answer is A\n\n'
        'Q: A train travels 180 miles in 3 hours. What is its average speed in mph?\n'
        'Options: A) 36  B) 60  C) 90  D) 120  E) 540\n'
        'A: Check: 180/3=60. Correct option B.\n'
        'The answer is B'
    ),
}


@PromptSetRegistry.register('aqua')
class AQUAPromptSet(PromptSet):

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
    def get_baseline_constraint(prompt_style: str, role: str | None = None) -> str:
        prompt_style = prompt_style.lower()
        base = """
I will ask you a multiple-choice math question.
There are 5 answer options enumerated as A, B, C, D, and E.
Only one option is correct.
Always select the best available option from A, B, C, D, and E.
"""
        if prompt_style == "vanilla":
            return base + """
Reply with only one letter: A, B, C, D, or E.
Do not include any analysis.
"""
        if prompt_style == "cot":
            return base + """
Reason step by step before choosing the answer.
Use at most 5 short sentences.
Do not write tables, exhaustive cases, or long derivations.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, D, or E.
"""
        if prompt_style == "complex_cot":
            return base + """
Reason carefully through the question before choosing the answer.
Compare the answer choices, eliminate incorrect options, and keep the analysis concise.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, D, or E.
"""
        if prompt_style == "php":
            return base + """
Provide concise progressive hints before choosing the answer.
Start from the key equation or concept, then narrow down the choices, then explain why the selected option is correct.
Use at most 5 short sentences.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, D, or E.
"""
        raise ValueError(f"Unsupported AQuA baseline prompt style: {prompt_style}")

    @staticmethod
    def get_baseline_answer_prompt(question, prompt_style: str):
        return f"The task is:\n\n{question}"

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a multiple-choice math problem with options A, B, C, D, and E. "
            "Only one is correct. "
            "Your response must be exactly one letter: A, B, C, D, or E. "
            "The first line of your reply must contain only the letter of your chosen answer, nothing else."
        )

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker for multiple-choice math problems."

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
"3. Provide an incorrect or misleading solution with a plausible but wrong answer.\n"
"4. Your reply must be less than 100 words.\n"
"5. The first line of your reply must contain only one letter: A, B, C, D, or E.\n"
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
    def postprocess_answer(answer: Union[str, List[str]], options=None) -> str:
        return extract_choice_answer(answer, choices=("A", "B", "C", "D", "E"))
