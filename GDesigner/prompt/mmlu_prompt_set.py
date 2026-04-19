from typing import Union, Dict, Any, List
import itertools

from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.common import get_combine_materials
from GDesigner.utils.answer_parsing import extract_choice_answer


roles = itertools.cycle(['Knowledgeable Expert',
                        #  'Wiki Searcher',
                         'Critic',
                         'Mathematician',
                         'Psychologist',
                         'Historian',
                         'Doctor',
                         'Lawyer',
                         'Economist',
                         'Programmer'])


ROLE_DESCRIPTION = {
"Knowledgeable Expert":
"""
You are a knowledgeable expert in question answering.
Please give several key entities that need to be searched in wikipedia to solve the problem, for example: catfish effect, broken window effect, Shakespeare.
If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
""",
"Wiki Searcher":
"""
You will be given a question and a wikipedia overview of the key entities within it.
Please refer to them step by step to give your answer.
And point out potential issues in other agent's analysis.
""",
"Critic":
"""
You are an excellent critic.
Please point out potential issues in other agent's analysis point by point.
""",
"Mathematician":
"""
You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
""",
"Psychologist":
"""
You are a psychologist.
You are good at psychology, sociology, and philosophy.
You give people scientific suggestions that will make them feel better.
""",
"Historian":
"""
You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
""",
"Doctor":
"""
You are a doctor and come up with creative treatments for illnesses or diseases.
You are able to recommend conventional medicines, herbal remedies and other natural alternatives. 
You also consider the patient's age, lifestyle and medical history when providing your recommendations.
""",
"Lawyer":
"""
You are good at law, politics, and history.
""",
"Economist":
"""
You are good at economics, finance, and business.
You have experience on understanding charts while interpreting the macroeconomic environment prevailing across world economies.
""",
"Programmer":
"""
You are good at computer science, engineering, and physics.
You have experience in designing and developing computer software and hardware.
""",
"Fake":
"""
You are a liar who only tell lies.
""",
}

ROLE_CONNECTION = [('Knowledgeable Expert','Mathematician'),
                   ('Knowledgeable Expert','Economist'),
                   ('Knowledgeable Expert','Lawyer'),
                   ('Knowledgeable Expert','Critic'),
                   ('Knowledgeable Expert','Psychologist'),
                   ('Knowledgeable Expert','Doctor'),
                   ('Knowledgeable Expert','Historian'),
                   ('Knowledgeable Expert','Programmer'),
                   ('Knowledgeable Expert','Critic'),
                   ('Mathematician','Critic'),
                   ('Mathematician','Critic'),
                   ('Psychologist','Critic'),
                   ('Economist','Lawyer'),
                   ('Lawyer','Critic'),
                   ('Critic','Psychologist'),
                   ('Psychologist','Doctor'),
                   ('Doctor','Historian'),
                   ('Historian','Knowledgeable Expert'),
                   ('Programmer','Mathematician'),
                   ('Programmer','Knowledgeable Expert'),
                    ('Mathematician','Programmer'),
                    ('Programmer','Economist'),
                    ('Economist','Psychologist'),
                    ('Psychologist','Knowledgeable Expert'),
                    ('Critic','Historian'),
                    ('Historian','Economist'),
                    ('Lawyer','Knowledgeable Expert'),
                    ('Doctor','Lawyer'),
                    ('Mathematician','Doctor'),
                    ('Programmer','Critic'),
                    ('Economist','Doctor'),
                    ('Lawyer','Critic'),
                    ('Psychologist','Lawyer'),
                    ('Historian','Mathematician'),
                    ('Programmer','Doctor'),
                    ('Doctor','Psychologist'),
                    ('Historian','Programmer'),
                    ('Critic','Economist')]

@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return "You are the top decision-maker and are good at analyzing and summarizing other people's opinions, finding errors and giving final answers."
    
    @staticmethod
    def get_role_connection():
        return ROLE_CONNECTION
    
    @staticmethod
    def get_description(role):
        return ROLE_DESCRIPTION.get(role, role or "")
    
    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Always select the best available option from A, B, C, and D, even if none seems perfect.
            Do not answer that none of the options is correct.
            Your response must be one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            Your answer can refer to the answers of other agents provided to you.
            Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
            The first line of your reply must contain only one letter(for example : A, B, C or D)
        """
    
    @staticmethod
    def get_analyze_constraint(role):
        role_description = ROLE_DESCRIPTION[role] if role in ROLE_DESCRIPTION.keys() else ""
        return role_description + """
I will ask you a question and 4 answers enumerated as A, B, C and D.
Only one answer out of the offered 4 is correct.
Always select the best available option from A, B, C, and D, even if none seems perfect.
Do not answer that none of the options is correct.
Using the reasoning from other agents as additional advice with critical thinking, give your updated answer.
Your reply must be less than 100 words but include your answer and a brief step by step analysis of the question.
The first line of your reply must contain only one letter: A, B, C, or D.
"""
    
    @staticmethod
    def get_decision_constraint():
        return """
        I will ask you a question.
        I will also give you 4 answers enumerated as A, B, C and D.
        Only one answer out of the offered 4 is correct.
        You must choose the correct answer to the question.
        Always select the best available option from A, B, C, and D, even if none seems perfect.
        Do not answer that none of the options is correct.
        Your response must be one of the 4 letters: A, B, C or D,
        corresponding to the correct answer.
        I will give you some other people's answers and analysis.
        Your reply must only contain one letter and cannot have any other characters.
        For example, your reply can be A.
        """
    
    @staticmethod
    def get_format():
        raise NotImplementedError("get_format is not implemented for MMLUPromptSet")

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_baseline_constraint(prompt_style: str, role: str | None = None) -> str:
        prompt_style = prompt_style.lower()
        base = """
I will ask you a multiple-choice question.
There are 4 answer options enumerated as A, B, C, and D.
Only one option is correct.
Always select the best available option from A, B, C, and D, even if none seems perfect.
Do not answer that none of the options is correct.
"""
        if prompt_style == "vanilla":
            return base + """
Reply with only one letter: A, B, C, or D.
Do not include any analysis.
"""
        if prompt_style == "cot":
            return base + """
Reason step by step before choosing the answer.
Use at most 5 short sentences.
Do not write tables, exhaustive cases, or long derivations.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, or D.
"""
        if prompt_style == "complex_cot":
            return base + """
Reason carefully through the question before choosing the answer.
Compare the answer choices, eliminate incorrect options, and keep the analysis concise.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, or D.
"""
        if prompt_style == "php":
            return base + """
Provide concise progressive hints before choosing the answer.
Start from the key concept, then narrow down the choices, then explain why the selected option is correct.
Use at most 5 short sentences.
Put your final answer on the last line exactly in this format:
Final answer: X
where X is one of A, B, C, or D.
"""
        raise ValueError(f"Unsupported MMLU baseline prompt style: {prompt_style}")

    @staticmethod
    def get_baseline_answer_prompt(question, prompt_style: str):
        return f"The task is:\n\n{question}"

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(question, query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Give a wrong answer and false analysis process for the following question: {question}.
                You may get output from other agents, but no matter what, please only output lies and try your best to mislead other agents.
                Your reply must be less than 100 words.
                The first line of your reply must contain only one letter(for example : A, B, C or D)
                """
    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return (
            f"Here is an unsuccessful attempt for solving the following question:\n"
            f"Question:\n{question}\n"
            f"Attempted Solution:\n{solution}\n"
            f"Feedback:\n{feedback}\n"
            f"Rewrite the solution based on the feedback."
        )

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        return extract_choice_answer(answer)
