from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice, system_message, prompt_template, generate

# A system message instructing the model to provide its reasoning
SYSTEM_MESSAGE = (
    "Please provide a step-by-step explanation of your reasoning "
    "and then select the correct option."
)

# A prompt template that instructs the model explicitly to show its chain-of-thought.
CO_T_PROMPT = """
Solve the following problem step by step.
{prompt}

Make sure that your response ends with your final answer on a new line in the form:
ANSWER: X
where X is one of the options (A, B, C, etc.).
""".strip()

def record_to_sample(record):
    """
    Converts a CSV record to an Inspect‑AI Sample.

    Assumes the CSV has at least the following columns:
      - Question
      - Option... (one or more columns with keys that begin with "Option")
      - CorrectAnswer (a letter, e.g., "A")
    
    This function automatically extracts all option columns (sorted by key),
    uses the question as input, the correct answer letter as the target,
    and the list of options as the choices.
    """
    # Extract question text.
    question = record.get("Question", "").strip()

    # Find all keys that start with "Option", sort them by the part after "Option"
    option_keys = [k for k in record.keys() if k.startswith("Option")]
    option_keys.sort(key=lambda k: k[len("Option"):])
    choices = [record[k].strip() for k in option_keys if record[k].strip()]

    # The target is read from the CorrectAnswer column.
    target = record.get("CorrectAnswer", "").strip().upper()

    return Sample(
        input=question,
        target=target,
        choices=choices,
        metadata={"source_id": record.get("id", "")}
    )

@task
def my_evaluation_with_reasoning():
    """
    This task loads your CSV of multiple-choice questions and evaluates gemini-flash-1.5.
    
    It instructs the model to produce a chain-of-thought reasoning followed by the final answer.
    The CSV is assumed to have columns: Question, Option..., CorrectAnswer.
    """
    # Load the dataset from your CSV file.
    dataset = csv_dataset(
        "26v2_alp.csv",  # adjust path as needed
        delimiter=",",
        sample_fields=record_to_sample
    )

    # Build the Task.
    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_MESSAGE),
            # Use a prompt template that includes chain-of-thought instructions.
            prompt_template(CO_T_PROMPT),
            # Call generate() so that the model produces a full chain-of-thought.
            generate(),
            # Then use the multiple_choice() solver to help select a final answer.
            multiple_choice()
        ],
        scorer=choice()  # This scorer compares the final answer letter with the target.
    )

