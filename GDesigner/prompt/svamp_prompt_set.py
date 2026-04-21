"""
PromptSet for SVAMP dataset.
Word-problem arithmetic reasoning (single-hop and multi-hop).
"""
from typing import Dict, Any, Union, List
import itertools

from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.common import get_combine_materials
from datasets.svamp_dataset import svamp_postprocess_answer


roles = itertools.cycle([
    'Mathematical Analyst',
    'Math Solver',
    'Programming Expert',
    'Inspector',
])


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


# Few-shot examples sampled from SVAMP dataset
FEW_SHOT_DATA = {
    "Mathematical Analyst": (
        'Q: Sam has 3 boxes of crayons. Each box has 12 crayons. He gives 8 crayons to his friend. How many crayons does Sam have?\n'
        'A:\n'
        'Step 1: 3 boxes × 12 crayons = 36 crayons total.\n'
        'Step 2: 36 - 8 = 28 crayons.\n'
        'The answer is 28\n\n'
        'Q: A baker made 240 cupcakes. He sold 85 on Monday and 73 on Tuesday. How many cupcakes are left?\n'
        'A:\n'
        'Step 1: 85 + 73 = 158 cupcakes sold.\n'
        'Step 2: 240 - 158 = 82 cupcakes left.\n'
        'The answer is 82'
    ),
    "Math Solver": (
        'Q: Sam has 3 boxes of crayons. Each box has 12 crayons. He gives 8 crayons to his friend. How many crayons does Sam have?\n'
        'A:\n'
        'Hint: The answer is near to 28.\n'
        'Step 1: 3 × 12 = 36. Step 2: 36 - 8 = 28.\n'
        'The answer is 28\n\n'
        'Q: A baker made 240 cupcakes. He sold 85 on Monday and 73 on Tuesday. How many cupcakes are left?\n'
        'A:\n'
        'Hint: The answer is near to 82.\n'
        'Step 1: 85 + 73 = 158. Step 2: 240 - 158 = 82.\n'
        'The answer is 82'
    ),
    "Programming Expert": (
        'Q: Sam has 3 boxes of crayons. Each box has 12 crayons. He gives 8 crayons to his friend. How many crayons does Sam have?\n'
        'A:\n'
        '```python\n'
        'def solve():\n'
        '    return 3 * 12 - 8\n'
        '\n'
        'print(solve())\n'
        '```\n\n'
        'Q: A baker made 240 cupcakes. He sold 85 on Monday and 73 on Tuesday. How many cupcakes are left?\n'
        'A:\n'
        '```python\n'
        'def solve():\n'
        '    return 240 - 85 - 73\n'
        '\n'
        'print(solve())\n'
        '```'
    ),
    "Inspector": (
        'Q: Sam has 3 boxes of crayons. Each box has 12 crayons. He gives 8 crayons to his friend. How many crayons does Sam have?\n'
        'A:\n'
        'Check: 3×12=36, 36-8=28. Correct.\n'
        'The answer is 28\n\n'
        'Q: A baker made 240 cupcakes. He sold 85 on Monday and 73 on Tuesday. How many cupcakes are left?\n'
        'A:\n'
        'Check: 85+73=158, 240-158=82. Correct.\n'
        'The answer is 82'
    ),
}


@PromptSetRegistry.register('svamp')
class SVAMPPromptSet(PromptSet):

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
I will ask you an arithmetic word problem.
Solve the problem and provide the numeric answer without units.
Always put your final answer on the last line exactly in this format:
The answer is X
where X is the numeric answer.
"""
        if prompt_style == "vanilla":
            return base + """
Reply with only the final answer line.
Do not include analysis.
"""
        if prompt_style == "cot":
            return base + """
Reason step by step before giving the final answer.
Keep the reasoning concise.
"""
        if prompt_style == "complex_cot":
            return base + """
Reason carefully through the quantities, operations, and edge cases before giving the final answer.
Keep the reasoning concise and avoid unnecessary alternatives.
"""
        if prompt_style == "php":
            return base + """
Provide concise progressive hints before giving the final answer.
Start from the key quantity, then derive the needed intermediate values, then finish with the final answer line.
"""
        raise ValueError(f"Unsupported SVAMP baseline prompt style: {prompt_style}")

    @staticmethod
    def get_baseline_answer_prompt(question, prompt_style: str):
        return f"The task is:\n\n{question}"

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math word problem and solutions from other agents. "
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
        return svamp_postprocess_answer(answer)
