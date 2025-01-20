import csv
import random
import statistics
from decimal import Decimal

def all_options_are_integers(options):
    """
    Returns True if every value in 'options' is effectively an integer,
    e.g., 3.0 -> True, 3.1 -> False.
    """
    return all(x.is_integer() for x in options)

def get_max_decimal_places(values):
    """
    For a list of floats, returns the maximum number of decimal places among them.
    E.g. [5.0, 3.141, 3.14] -> 3 because 3.141 has 3 decimals.
    """
    max_dp = 0
    for v in values:
        # Convert to string, then to a Decimal to inspect its exponent
        dec = Decimal(str(v)).normalize()
        # dec.as_tuple().exponent is negative if there's a decimal part.
        # E.g. "3.141" -> exponent = -3 => 3 decimal places
        dp = max(0, -dec.as_tuple().exponent)
        if dp > max_dp:
            max_dp = dp
    return max_dp

def read_numeric_questions_from_csv(csv_file_path):
    """
    Reads a CSV of questions where each row has:
       Question, OptionA, OptionB, OptionC, OptionD, CorrectAnswer

    - OptionA/B/C/D are numeric strings (e.g., "3.14", "10")
    - CorrectAnswer is one of "A", "B", "C", or "D"

    Returns a list of dicts:
       {
         'question': <str>,
         'options': [float, float, float, float],
         'correct_answer': float
       }
    """
    questions_data = []
    
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Parse the four options as floats
            option_a = float(row['OptionA'])
            option_b = float(row['OptionB'])
            option_c = float(row['OptionC'])
            option_d = float(row['OptionD'])
            
            # CorrectAnswer column contains "A", "B", "C", or "D"
            correct_answer_letter = row['CorrectAnswer'].strip().upper()
            
            # Determine the correct numeric value
            if correct_answer_letter == "A":
                correct_ans = option_a
            elif correct_answer_letter == "B":
                correct_ans = option_b
            elif correct_answer_letter == "C":
                correct_ans = option_c
            elif correct_answer_letter == "D":
                correct_ans = option_d
            else:
                raise ValueError(
                    f"CorrectAnswer must be one of 'A', 'B', 'C', or 'D'. "
                    f"Got '{correct_answer_letter}' instead."
                )
            
            question_entry = {
                'question': row['Question'],
                'options': [option_a, option_b, option_c, option_d],
                'correct_answer': correct_ans
            }
            questions_data.append(question_entry)
    
    return questions_data

def generate_gaussian_distractors(all_4_options, correct_answer, how_many):
    """
    Generates 'how_many' distractors via a Gaussian distribution whose mean and stdev
    are computed from the 4 original options (including the correct answer).

    Key rules:
      - Skip negative values (candidate < 0).
      - If all original options are integers, round final distractors to int.
      - Otherwise, keep them as float.
      - Skip if candidate == correct_answer.
      - It is OK if candidate == one of the original distractors (besides correct answer).
    """
    # Check if we want integer-only or float
    integer_only = all_options_are_integers(all_4_options)

    mean = statistics.mean(all_4_options)
    unique_values = set(all_4_options)
    
    # If all 4 are the same, stdev = 0.0; fallback to stdev=1.0
    if len(unique_values) > 1:
        stdev = statistics.stdev(all_4_options)
    else:
        stdev = 0.0
    if stdev == 0.0:
        stdev = 1.0

    distractors = set()
    
    while len(distractors) < how_many:
        candidate = random.gauss(mean, stdev)

        # Skip negative values
        if candidate < 0:
            continue
        
        # If we want integers, round
        if integer_only:
            candidate = int(round(candidate))
        
        # Skip if it's exactly the correct answer
        if candidate == correct_answer:
            continue
        
        # It's now OK if candidate matches an original distractor or other original option
        # so we do NOT skip those.
        
        distractors.add(candidate)
    
    return list(distractors)

def expand_numeric_answers_for_question(question_data, N):
    """
    Returns a list of N answers (formatted as strings):
      1) The correct answer from the CSV
      2) (N-1) newly generated Gaussian distractors (not automatically including any original distractors)
    """
    correct_answer = question_data['correct_answer']
    original_options = question_data['options']  # includes correct answer

    # Decide how many distractors we need
    M = N - 1

    # Generate all M distractors from Gaussian distribution
    # (No original distractors are automatically included.)
    new_distractors = generate_gaussian_distractors(
        all_4_options=original_options,
        correct_answer=correct_answer,
        how_many=M
    )

    # Combine correct answer + newly generated distractors
    all_answers = [correct_answer] + new_distractors
    random.shuffle(all_answers)

    # Determine if we should format everything as integers or floats
    if not all_options_are_integers(original_options):
        # If at least one original option had decimals, find the max decimal places
        max_decimals = get_max_decimal_places(original_options)
        # Format all answers to that decimal precision
        all_answers_formatted = [f"{val:.{max_decimals}f}" for val in all_answers]
    else:
        # Otherwise, keep them as integers
        all_answers_formatted = [str(int(val)) for val in all_answers]
    
    return all_answers_formatted

def generate_expanded_quiz_numeric(csv_file_path, N):
    """
    1) Reads numeric questions from CSV.
    2) For each question, we produce N answers: 
       - The correct answer
       - (N-1) newly generated Gaussian distractors
    3) Returns a list of dicts with final formatted strings:
       {
         'question': <str>,
         'correct_answer': <str>,
         'answers': [<str>, <str>, ...]
       }
    """
    questions_data = read_numeric_questions_from_csv(csv_file_path)
    
    expanded_questions = []
    for q in questions_data:
        expanded_answers = expand_numeric_answers_for_question(q, N)
        
        # Format the correct answer the same way we formatted the rest
        # If the question is integer-based, correct_answer is an int
        # Otherwise, correct_answer is float with max_decimals from original
        if not all_options_are_integers(q['options']):
            max_decimals = get_max_decimal_places(q['options'])
            correct_answer_formatted = f"{q['correct_answer']:.{max_decimals}f}"
        else:
            correct_answer_formatted = str(int(q['correct_answer']))
        
        expanded_questions.append({
            'question': q['question'],
            'correct_answer': correct_answer_formatted,
            'answers': expanded_answers  # This includes the correct answer in random position
        })
    
    return expanded_questions

def write_expanded_numeric_questions_to_csv(expanded_questions, output_file_path):
    """
    Writes the expanded questions to a CSV with columns in this order:
       question, answer_1, answer_2, ..., answer_n, correct_answer
    """
    import csv
    
    # Determine the maximum number of answers in any question
    max_answers = max(len(item['answers']) for item in expanded_questions)
    
    # We want:
    #   question  | answer_1 | ... | answer_n | correct_answer
    fieldnames = (
        ["question"] + 
        [f"answer_{i+1}" for i in range(max_answers)] +
        ["correct_answer"]
    )
    
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in expanded_questions:
            row_data = {}
            row_data["question"] = item["question"]
            
            # Fill in each answer (these are the distractors + correct answer in random order)
            for i, ans in enumerate(item["answers"]):
                row_data[f"answer_{i+1}"] = ans
            
            # Lastly, put the correct answer in the final column
            row_data["correct_answer"] = item["correct_answer"]
            
            writer.writerow(row_data)

def main():
    input_csv = 'Physics Test Questions.csv'
    output_csv = 'modified_test_questions.csv'
    desired_num_answers = 10  # e.g., 1 correct + 5 distractors

    expanded_data = generate_expanded_quiz_numeric(input_csv, desired_num_answers)
    write_expanded_numeric_questions_to_csv(expanded_data, output_csv)

if __name__ == "__main__":
    main()


