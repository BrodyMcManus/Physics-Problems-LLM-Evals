import re
from inspect_ai import Task, task, eval
from inspect_ai.dataset import csv_dataset, Sample
from inspect_ai.solver import solver
from inspect_ai.scorer import Score, CORRECT, INCORRECT, scorer, accuracy, stderr

# -------------------------------------------------------------------------------
def record_to_sample(record: dict) -> Sample:
    """
    Convert a CSV row (a dict) into an Inspect Sample.

    Expected CSV header columns:
      - "Question": The question text.
      - "Option..." columns: All answer choices (e.g. OptionAA, OptionAB, …).
      - "CorrectAnswer": The correct answer label as recorded (e.g. "BK").

    This function extracts the labels from the header (by removing "Option")
    and stores them in sample.metadata["option_labels"]. It also removes any
    pre-existing label from the cell text.
    """
    question = record["Question"]
    # Get all option keys and sort them.
    option_keys = sorted([key for key in record.keys() if key.startswith("Option")])
    # Extract the labels from the header (e.g. "OptionAA" -> "AA")
    labels = [key[len("Option"):] for key in option_keys]

    raw_choices = [record[k].strip() for k in option_keys]
    cleaned_choices = []
    # Remove any leading label pattern from the cell text (e.g. "A) 400" -> "400")
    pattern = r"^\s*[A-Za-z]{1,3}\)\s*(.*)$"
    for choice in raw_choices:
        m = re.match(pattern, choice)
        if m:
            cleaned_choices.append(m.group(1).strip())
        else:
            cleaned_choices.append(choice)
    
    correct = record["CorrectAnswer"].strip()
    sample = Sample(input=question, choices=cleaned_choices, target=correct)
    sample.metadata = {"option_labels": labels}
    return sample

# -------------------------------------------------------------------------------
@scorer(metrics=[accuracy(), stderr()])
def simple_choice_scorer():
    """
    A simple scorer that extracts the final answer from the model output.
    It expects the model output to include a line in the format:
    
       ANSWER: <option label>
       
    The extracted option label is compared directly to the sample target.
    """
    async def score(state, target):
        output = state.output.completion.strip()
        pattern = r"ANSWER:\s*(?P<ans>[A-Z]+)"
        match = re.search(pattern, output)
        if match:
            final_ans = match.group("ans").strip()
        else:
            tokens = output.split()
            final_ans = tokens[-1] if tokens else ""
        is_correct = (final_ans == target.text)
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=final_ans,
            explanation=output
        )
    return score

# -------------------------------------------------------------------------------
@solver
def custom_multiple_choice_no_cot(template: str = None):
    """
    A custom multiple-choice solver that:
      - Uses the question and the cleaned choices from the sample.
      - Uses the option labels stored in sample.metadata["option_labels"] to build the prompt.
      - Instructs the model to output its final answer in the following format:
      
          ANSWER: <option label>
          
      with no request for chain-of-thought reasoning.
    """
    async def solve(state, generate):
        # Retrieve the question (assumed to be in the user prompt text)
        question = state.user_prompt.text
        choices = state.choices
        labels = state.metadata.get("option_labels")
        if not labels or len(labels) != len(choices):
            labels = [chr(65 + i) for i in range(len(choices))]
        
        # Convert each choice to a string (if not already)
        def format_choice(choice):
            return choice.value if hasattr(choice, "value") else str(choice)
        
        formatted_choices = "\n".join(
            f"{label}) {format_choice(ch)}" for label, ch in zip(labels, choices)
        )
        
        if template is None:
            prompt = (
                f"Question:\n{question}\n\n"
                "Please output your final answer on its own line in the following format:\n"
                "ANSWER: <option label>\n"
                "Note: The final answer must be exactly one of the provided option labels.\n"
                f"Choices:\n{formatted_choices}\n"
            )
        else:
            prompt = template.format(question=question, choices=formatted_choices)
        
        # Update the user prompt text so the model sees this new prompt.
        state.user_prompt.text = prompt
        return await generate(state)
    return solve

# -------------------------------------------------------------------------------
@task
def custom_mc_no_cot_eval(csv_file_path: str):
    dataset = csv_dataset(csv_file_path, sample_fields=record_to_sample)
    return Task(
        dataset=dataset,
        solver=[ custom_multiple_choice_no_cot() ],
        scorer=simple_choice_scorer()
    )

# -------------------------------------------------------------------------------
#if __name__ == "__main__":
#    eval(custom_mc_no_cot_eval("Mod_MMLU4.csv"), model="google/gemini-1.5-flash")