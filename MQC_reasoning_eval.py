# custom_mc_eval.py
#
# This script defines an Inspect task that reads multiple-choice questions
# from a CSV file and evaluates the Gemini 1.5 Flash model.
#
# The CSV file is expected to have a header with columns like:
#
#   Question,OptionAA,OptionAB,OptionAC,...,OptionDV,CorrectAnswer
#
# The CorrectAnswer column holds the answer label exactly as in the header
# (for example, "BK"). This custom solution extracts those labels from the
# header and uses them when formatting the prompt.
#
# To run:
#   pip install inspect-ai
#   inspect eval custom_mc_eval.py --model google/gemini-1.5-flash
#

import re
from inspect_ai import Task, task, eval
from inspect_ai.dataset import csv_dataset, Sample
from inspect_ai.solver import solver, chain_of_thought
from inspect_ai.scorer import Score, CORRECT, INCORRECT, scorer, accuracy, stderr

# -------------------------------------------------------------------------------
def record_to_sample(record: dict) -> Sample:
    """
    Convert a CSV row (a dict) into an Inspect Sample.
    
    Expected CSV header columns:
      - "Question": The question text.
      - "Option..." columns: All answer choices (e.g. OptionAA, OptionAB, …).
      - "CorrectAnswer": The correct answer label as recorded (e.g. "BK").
    
    This function extracts the labels from the CSV header (by removing the
    "Option" prefix) and stores them in sample.metadata["option_labels"].
    It also cleans each cell by removing any pre-existing label from its text.
    """
    question = record["Question"]
    # Get all option column keys and sort them.
    option_keys = sorted([key for key in record.keys() if key.startswith("Option")])
    # Extract labels from the header, e.g. "OptionAA" -> "AA"
    labels = [key[len("Option"):] for key in option_keys]
    
    raw_choices = [record[key].strip() for key in option_keys]
    cleaned_choices = []
    # Remove any leading label pattern from the cell text.
    pattern = r"^\s*[A-Za-z]{1,3}\)\s*(.*)$"
    for choice in raw_choices:
        m = re.match(pattern, choice)
        if m:
            cleaned_choices.append(m.group(1).strip())
        else:
            cleaned_choices.append(choice)
    
    correct = record["CorrectAnswer"].strip()
    sample = Sample(input=question, choices=cleaned_choices, target=correct)
    # Store the extracted labels in metadata for our custom solver.
    sample.metadata = {"option_labels": labels}
    return sample

# -------------------------------------------------------------------------------
@scorer(metrics=[accuracy(), stderr()])
def choice_with_cot():
    """
    Custom scorer that extracts the chain-of-thought and final answer from the model output.
    Expects output in the format:
      Chain-of-thought:
      <your reasoning here>
      ANSWER: <option label>
    """
    async def score(state, target):
        output = state.output.completion.strip()
        pattern = r"Chain-of-thought:\s*(?P<cot>.*?)\s*[\r\n]+ANSWER:\s*(?P<ans>[A-Z]+)"
        match = re.search(pattern, output, re.DOTALL)
        if match:
            final_ans = match.group("ans").strip()
            cot_text = match.group("cot").strip()
        else:
            tokens = output.split()
            final_ans = tokens[-1] if tokens else ""
            cot_text = ""
        is_correct = (final_ans == target.text)
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=final_ans,
            explanation=f"Chain-of-thought:\n{cot_text}\n\nFull output:\n{output}"
        )
    return score

# -------------------------------------------------------------------------------
@solver
def custom_multiple_choice(template: str = None, cot: bool = True):
    """
    Custom multiple-choice solver that uses the option labels stored in the sample's metadata.
    It builds the choices string using the labels extracted from the CSV header.
    
    Instead of using auto-generated labels, it uses the labels from sample.metadata["option_labels"].
    If a custom template is not provided, a default prompt is constructed.
    """
    async def solve(state, generate):
        # Access the original question from the user prompt text.
        question = state.user_prompt.text  # This is our original question text.
        choices = state.choices           # These should be the cleaned choice strings.
        labels = state.metadata.get("option_labels")
        if not labels or len(labels) != len(choices):
            # Fallback: generate labels A, B, C, ... if metadata is missing.
            labels = [chr(65 + i) for i in range(len(choices))]
        
        # Convert each choice into a simple string.
        def format_choice(choice):
            return choice.value if hasattr(choice, "value") else str(choice)
        
        # Build the choices string using the labels from the CSV.
        formatted_choices = "\n".join(f"{label}) {format_choice(choice)}" 
                                       for label, choice in zip(labels, choices))
        
        if template is None:
            prompt = (
                f"Question: {question}\n\n"
                "Please provide your reasoning step-by-step. At the end, on a new line, output your final answer in the following format:\n"
                "Chain-of-thought:\n<your reasoning here>\nANSWER: <option label>\n"
                "Note: The final answer must be exactly one of the provided option labels.\n"
                f"Choices:\n{formatted_choices}\n"
            )
        else:
            prompt = template.format(question=question, choices=formatted_choices)
        
        # Update the first user message with our constructed prompt.
        state.user_prompt.text = prompt
        return await generate(state)
    return solve

# -------------------------------------------------------------------------------
@task
def custom_mc_csv_eval():
    """
    Define a task that:
      - Loads multiple-choice questions from a CSV file.
      - Uses chain-of-thought prompting so that the model outputs its reasoning.
      - Uses our custom multiple-choice solver that labels options based on the CSV header.
      - Grades the answer using our custom scorer that extracts the chain-of-thought.
    """
    dataset = csv_dataset("Mod_MMLU4.csv", sample_fields=record_to_sample)
    
    return Task(
        dataset=dataset,
        solver=[
            custom_multiple_choice(cot=True)
        ],
        scorer=choice_with_cot()
    )

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Run the evaluation on the Gemini 1.5 Flash model.
    eval(custom_mc_csv_eval(), model="google/gemini-1.5-flash")
