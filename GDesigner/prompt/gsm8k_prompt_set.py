from typing import Dict, Any, List, Union
import itertools
from GDesigner.prompt.prompt_set import PromptSet
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.prompt.common import get_combine_materials
from datasets.gsm8k_dataset import gsm8k_postprocess_answer

roles = itertools.cycle(['Math Solver',
                         'Mathematical Analyst',
                         'Programming Expert',
                         'Inspector',])

ROLE_DESCRIPTION = {
    "Math Solver":
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140.\n"
        "You will be given some examples you may refer to.",
    "Mathematical Analyst":
        "You are a mathematical analyst. "
        "You are good at arithmetic calculation, algebraic reasoning, and equation solving. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140.\n"
        "You will be given some examples you may refer to.",
    "Programming Expert":
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the (answer) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to.",
    "Inspector":
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140.\n"
        "You will be given some examples you may refer to.",
}

ROLE_CONNECTION = [
    ('Mathematical Analyst', 'Math Solver'),
    ('Mathematical Analyst', 'Programming Expert'),
    ('Mathematical Analyst', 'Inspector'),
    ('Math Solver', 'Programming Expert'),
    ('Programming Expert', 'Math Solver'),
    ('Programming Expert', 'Inspector'),
    ('Inspector', 'Math Solver'),
    ('Inspector', 'Programming Expert'),
    ('Inspector', 'Mathematical Analyst'),
]


# This function is inspired by/derived from the implementation in the following GitHub repository:
# Repository: https://github.com/chuanyang-Zheng/Progressive-Hint/blob/main/prompt/complex/complex_PHP_gsm8k.txt
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/tora/gsm8k.md
# Repository: https://github.com/microsoft/ToRA/blob/213c1c995038c73fab10343814df7a42f990f026/src/prompts/cot/gsm8k.md
FEW_SHOT_DATA = {
"Math Solver":
"""
Q: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week.
They have 2 chapters of their textbook to study and 4 worksheets to memorize.
They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet.
If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour,
include 3 10-minute snack breaks each day, and 30 minutes for lunch each day? (Hint: The answer is near to 4).

A: We know the Answer Hints: 4. With the Answer Hints: 4, we will answer the question.
Let's think step by step.
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break,
so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Q: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles? (Hint: The answer is near to 160, 145).
A: We know the Answer Hints: 160, 145. With the Answer Hints: 160, 145, we will answer the question.
Let's think step by step
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Q: Susy goes to a large school with 800 students, while Sarah goes to a smaller school with only 300 students. At the start of the school year, Susy had 100 social media followers. She gained 40 new followers in the first week of the school year, half that in the second week, and half of that in the third week. Sarah only had 50 social media followers at the start of the year, but she gained 90 new followers the first week, a third of that in the second week, and a third of that in the third week. After three weeks, how many social media followers did the girl with the most total followers have? (Hint: The answer is near to 180, 160).
A: We know the Answer Hints: 180, 160. With the Answer Hints: 180, 160, we will answer the question.
Let's think step by step
After one week, Susy has 100+40 = 140 followers.
In the second week, Susy gains 40/2 = 20 new followers.
In the third week, Susy gains 20/2 = 10 new followers.
In total, Susy finishes the three weeks with 140+20+10 = 170 total followers.
After one week, Sarah has 50+90 = 140 followers.
After the second week, Sarah gains 90/3 = 30 followers.
After the third week, Sarah gains 30/3 = 10 followers.
So, Sarah finishes the three weeks with 140+30+10 = 180 total followers.
Thus, Sarah is the girl with the most total followers with a total of 180.
The answer is 180
""",

"Mathematical Analyst":
"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: ## Problem solving process analysis

There are {ori_tree_num} trees originally.
Then there were {after_planted_tree_num} trees after some more were planted.
So the number of trees planted today {today_planted_num} is the number of trees after planting {after_planted_tree_num} minus the number of trees before planting {ori_tree_num}.
The answer is {today_planted_num} = {after_planted_tree_num} - {ori_tree_num}.

## Actual analysis and solution process

In this question, {ori_tree_num} = 15 and {after_planted_tree_num} = 21.
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So the number of trees planted today must have been 21 - 15 = 6.
The answer is 6

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A:## Problem solving process analysis

Originally, Leah had {Leah_num} Leah_num chocolates.
Her sister had {sister_num} chocolates.
So in total they had {all_num} = {Leah_num} + {sister_num} chocolates.
After eating {eating_num} chocolates, the number of chocolates they have left {remain_num} is {all_num} minus {eating_num}.
The answer is {remain_num} = {all_num} - {eating_num}.

## Actual analysis and solution process

In this question, {Leah_num} = 32, {sister_num} = 42 and {all_num} = 35.
So, in total they had 32 + 42 = 74 chocolates originally.
After eating 35 chocolates, they had 74 - 35 = 39 chocolates.
The answer is 39
""",

"Programming Expert":
"""
Q: Olivia has 23 dollars. She bought five bagels for 3 dollars each. How much money does she have left?
A:
```python\n
def money_left():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    remaining_money = money_initial - money_spent
    return remaining_money

answer = money_left()
\n```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A:
```python\n
def remaining_golf_balls():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    remaining_golf_balls = golf_balls_left
    return remaining_golf_balls

answer = remaining_golf_balls() 
\n```
""",

"Inspector":
"""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Step 1: 21 - 15 = 6 trees planted.
The answer is 6

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Step 1: 32 + 42 = 74 total. Step 2: 74 - 35 = 39 remaining.
The answer is 39
""",
}


@PromptSetRegistry.register('gsm8k')
class GSM8KPromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_constraint(role):
        return ROLE_DESCRIPTION.get(role, ROLE_DESCRIPTION["Math Solver"])

    @staticmethod
    def get_description(role):
        return ROLE_DESCRIPTION.get(role, role or "")

    @staticmethod
    def get_role_connection():
        return ROLE_CONNECTION

    @staticmethod
    def get_format():
        return "natural language"

    @staticmethod
    def get_answer_prompt(question, role="Mathematical Analyst"):
        shots = FEW_SHOT_DATA.get(role, FEW_SHOT_DATA["Mathematical Analyst"])
        return shots + "\n\n---END OF EXAMPLES---\n\nQ:" + question

    @staticmethod
    def get_baseline_constraint(prompt_style: str, role: str | None = None) -> str:
        prompt_style = prompt_style.lower()
        base = """
I will ask you a grade-school math word problem.
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
        raise ValueError(f"Unsupported GSM8K baseline prompt style: {prompt_style}")

    @staticmethod
    def get_baseline_answer_prompt(question, prompt_style: str):
        return f"The task is:\n\n{question}"

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a math problem, analysis and code from other agents. "
            "Please find the most reliable answer based on the analysis and results of other agents. "
            "Give reasons for making decisions. "
            "The last line of your output contains only the final result without any units, for example: The answer is 140"
        )

    @staticmethod
    def get_decision_role():
        return (
            "You are the top decision-maker. "
            "Good at analyzing and summarizing mathematical problems, judging and summarizing other people's solutions, and giving final answers to math problems."
        )

    @staticmethod
    def get_decision_few_shot():
        return """
Q:Josh decides to try flipping a house. He buys a house for 80000 dollars and then puts in 50000 dollars in repairs. This increased the value of the house by 150 percent. How much profit did he make?

A:After reviewing the analysis and results provided by the other agents, it appears that there is a discrepancy in the interpretation of the problem. The key point of contention is the base value on which the 150 percent increase is calculated.

Agents Agent_A, Agent_B, and Agent_C have calculated the 150 percent increase based on the total investment (purchase price plus repair costs), which is 130000 dollars. They have then added this increase to the total investment to find the new value of the house, and subsequently calculated the profit by subtracting the total investment from the new value of the house. This approach leads to a profit calculation of 195000 dollars.

Agent_D, however, has interpreted the problem differently. This agent suggests that the 150 percent increase should be calculated based on the initial purchase price only, not the total investment. Following this method, the increased value is calculated as 80000 dollars (initial purchase price) times 1.5, which equals 200000 dollars. The profit is then calculated by subtracting the total investment (130000 dollars) from this increased value, resulting in a profit of 70000 dollars.

The problem statement is ambiguous because it does not explicitly state whether the 150 percent increase is based on the initial purchase price alone or the total investment. However, the most common interpretation in real estate when referring to an increase in value due to repairs would be based on the initial purchase price, as the value increase is typically a reflection of the property market value appreciation, not the sum of costs incurred.

Therefore, based on the typical real estate valuation practice and the more common interpretation of such scenarios, Agent_D approach seems to be the most reliable. The profit should be calculated based on the increased value from the initial purchase price, not the total investment.

The final result, based on the most reliable interpretation, is a profit of 70000.

The answer is 70000
"""

    @staticmethod
    def get_react_prompt(question, solution, feedback):
        return (
            "Here is an unsuccessful attempt for solving the following question:\n"
            "Question:\n{}\n"
            "Attempted Solution:\n{}\n"
            "Feedback:\n{}\n"
            "Rewrite the code based on the feedback and the following question:\n{}"
        ).format(question, solution, feedback, question)

    @staticmethod
    def get_query_prompt(question):
        return (
            "# Information Gathering for Question Resolution\n\n"
            "Evaluate if additional information is needed to answer the question.\n"
            "If a web search or file analysis is necessary, outline specific clues or details to be searched for.\n\n"
            "## Target Question:\n{}\n\n"
            "## Clues for Investigation:\n"
            "Identify critical clues and concepts within the question that are essential for finding the answer."
        ).format(question)

    @staticmethod
    def get_file_analysis_prompt(query, file):
        return (
            "# File Analysis Task\n\n"
            "## Information Extraction Objective:\n---\n{}\n---\n\n"
            "## File Under Analysis:\n---\n{}\n---\n\n"
            "## Instructions:\n"
            "1. Identify the key sections in the file relevant to the query.\n"
            "2. Extract and summarize the necessary information from these sections.\n"
            "3. Ensure the response is focused and directly addresses the query.\n"
            "Example: 'Identify the main theme in the text.'"
        ).format(query, file)

    @staticmethod
    def get_websearch_prompt(question, query):
        return (
            "# Web Search Task\n\n"
            "## Original Question:\n---\n{}\n---\n\n"
            "## Targeted Search Objective:\n---\n{}\n---\n\n"
            "## Simplified Search Instructions:\n"
            "Generate three specific search queries directly related to the original question. Each query should focus on key terms from the question. Format the output as a comma-separated list.\n"
            "For example, if the question is 'Who will be the next US president?', your queries could be: 'US presidential candidates, current US president, next US president'.\n"
            "Remember to format the queries as 'query1, query2, query3'."
        ).format(question, query)

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return (
            "# Adversarial Challenge Task\n\n"
            "## Target Question:\n---\n{}\n---\n\n"
            "## Instructions:\n"
            "1. Identify potential weaknesses or edge cases in the question.\n"
            "2. Analyze the question from an adversarial perspective.\n"
            "3. Provide an incorrect or misleading solution.\n"
            "4. Write your full implementation using a Python code block:\n"
            + "```python\n"
            + "# Your implementation here\n"
            + "```\n"
            + "Do not include anything other than Python code blocks in your response."
        ).format(question)

    @staticmethod
    def get_distill_websearch_prompt(question, query, results):
        return (
            "# Summarization of Search Results\n\n"
            "## Original question:\n---\n{}\n---\n\n"
            "## Required Information for Summary:\n---\n{}\n---\n\n"
            "## Analyzed Search Results:\n---\n{}\n---\n\n"
            "## Instructions for Summarization:\n"
            "1. Review the provided search results and identify the most relevant information related to the question and query.\n"
            "2. Extract and highlight the key findings, facts, or data points from these results.\n"
            "3. Organize the summarized information in a coherent and logical manner.\n"
            "4. Ensure the summary is concise and directly addresses the query, avoiding extraneous details.\n"
            '5. If the information from web search is useless, directly answer: "No useful information from WebSearch".'
        ).format(question, query, results)

    @staticmethod
    def get_reflect_prompt(question, answer):
        return (
            "# Reflection on the Task\n\n"
            "## Reflection Question:\n---\n{}\n---\n\n"
            "## Your Previous Answer:\n---\n{}\n---\n\n"
            "## Instructions:\n"
            "Reflect on your answer process, considering the accuracy, method, and reasoning."
        ).format(question, answer)

    @staticmethod
    def get_self_consistency(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Self-Consistency Evaluation Task\n\n"
            "## Question for Review:\n---\n{}\n---\n\n"
            "## Reviewable Answers:\n---\n{}\n---\n\n"
            "## Instructions for Selection:\n"
            "1. Read each answer and assess how it addresses the question.\n"
            "2. Compare the answers for their adherence to the given question's criteria and logical coherence.\n"
            "3. Identify the answer that best aligns with the question's requirements and is the most logically consistent.\n"
            "4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
            "5. Copy the most suitable answer as it is, without modification, to maintain its original form.\n"
            "6. Adhere to the constraints: {}.\n"
            "Note: If no answer fully meets the criteria, choose and copy the one that is closest to the requirements."
        ).format(question, formatted_answers, constraint)

    @staticmethod
    def get_select_best(question: str, answers: list, constraint: str) -> str:
        formatted_answers = "\n".join([f"Answer {index + 1}: {answer}" for index, answer in enumerate(answers)])
        return (
            "# Best Answer Evaluation Task\n\n"
            "## Question:\n---\n{}\n---\n\n"
            "## Candidate Answers for Evaluation:\n---\n{}\n---\n\n"
            "## Evaluation Instructions:\n"
            "1. Examine the question closely to understand its requirements.\n"
            "2. Read each candidate answer thoroughly and assess its relevance and accuracy about the question.\n"
            "3. Choose the answer that most accurately and completely addresses the question.\n"
            "4. Ignore the candidate answers if they do not give a direct answer, for example, using 'unable to ...', 'as an AI ...'.\n"
            "5. Copy the chosen answer exactly as it is presented, maintaining its original format.\n"
            "6. Adhere to the constraints: {}.\n"
            "Note: If none of the answers fully meet the question's criteria, select the one closest to fulfilling them."
        ).format(question, formatted_answers, constraint)

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        return gsm8k_postprocess_answer(answer)
