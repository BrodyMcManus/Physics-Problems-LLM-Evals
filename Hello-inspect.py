from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import (               
  multiple_choice, system_message  
)                                             

SYSTEM_MESSAGE = """
You are a computer security expert and you are tasked with
providing concise response to the following questions.
Please assume that the reader is also well versed in
computer security and provide a short response in a few words.
"""

# @task
# def theory_of_mind():
#     return Task(
#         dataset=example_dataset("theory_of_mind"),
#         solver=[
#           prompt_template(DEFAULT_PROMPT),
#           generate(),
#           self_critique()
#         ],
#         scorer=model_graded_fact()
#     )
# @task
# def security_guide():
#     return Task(
#         dataset=example_dataset("security_guide"),
#         solver=[system_message(SYSTEM_MESSAGE), generate()],
#         scorer=model_graded_fact(),
#     )


SYSTEM_MESSAGE="""
Choose the most plausible continuation for the story.
"""
def record_to_sample(record):
    return Sample(
        input=record["ctx"],
        target=chr(ord("A") + int(record["label"])),
        choices=record["endings"],
        metadata=dict(
            source_id=record["source_id"]
        )
    )
    

@task
def hellaswag():
    
    # dataset
    dataset = hf_dataset(
        path="hellaswag",
        split="validation",
        sample_fields=record_to_sample,
        trust=True
    )

    return Task(
        dataset=dataset,
        solver=[system_message(SYSTEM_MESSAGE),multiple_choice()],
        scorer=choice(),
    )

