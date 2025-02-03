import csv
import random
import statistics
from decimal import Decimal
import math

###############################################################################
# 1. Reading & Basic Helpers
###############################################################################

def get_label_length(n):
    length = 1
    while 26**length < n:
        length += 1
    return length

def index_to_label(idx, length):
    chars = []
    for _ in range(length):
        digit = idx % 26
        char = chr(ord('A') + digit)
        chars.append(char)
        idx //= 26
    return ''.join(reversed(chars))

def generate_labels(n):
    length = get_label_length(n)
    return [index_to_label(i, length) for i in range(n)]


def read_numeric_questions_from_csv(csv_file_path):
    """
    Reads a CSV with columns:
       Question, OptionA, OptionB, OptionC, OptionD, CorrectAnswer
    where OptionA/B/C/D are numeric strings, and CorrectAnswer is
    one of "A","B","C","D". Returns a list of dicts:
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
            option_a = float(row['OptionA'])
            option_b = float(row['OptionB'])
            option_c = float(row['OptionC'])
            option_d = float(row['OptionD'])
            
            correct_letter = row['CorrectAnswer'].strip().upper()
            if correct_letter == "A":
                correct_ans = option_a
            elif correct_letter == "B":
                correct_ans = option_b
            elif correct_letter == "C":
                correct_ans = option_c
            elif correct_letter == "D":
                correct_ans = option_d
            else:
                raise ValueError(
                    f"CorrectAnswer must be A, B, C, or D. Got {correct_letter}"
                )
            
            questions_data.append({
                'question': row['Question'],
                'options': [option_a, option_b, option_c, option_d],
                'correct_answer': correct_ans
            })
    return questions_data

def generate_zero_answer_questions(csv_file_path):
    """
    Reads numeric questions from the CSV. For each question:
      1) Finds the max decimal places among the 4 original options.
      2) Appends ' Round your answer to X places' to the question text.
      3) Stores the question and correct answer in a minimal structure.
    Returns a list of dicts:
      [ { "question": <updated question>, "correct_answer": <str> }, ... ]
    """
    questions_data = read_numeric_questions_from_csv(csv_file_path)
    zero_answer_data = []
    
    for q in questions_data:
        original_options = q['options']
        correct_answer = q['correct_answer']
        
        # Find max decimal places
        max_dp = get_max_decimals_in_original(original_options)
        
        # Update the question text
        updated_question = q['question'] + f" Round your answer to {max_dp} decimal places."
        
        # We'll also format the correct answer to that many decimal places
        correct_str = f"{correct_answer:.{max_dp}f}"
        
        zero_answer_data.append({
            "question": updated_question,
            "correct_answer": correct_str
        })
    
    return zero_answer_data

def get_max_decimals_in_original(options):
    """
    Returns the max number of decimal places found among 'options'.
    E.g. [3.14, 2.0, 5.1234] => 4
    """
    max_dp = 0
    for val in options:
        dec = Decimal(str(val)).normalize()
        exponent = dec.as_tuple().exponent
        if exponent < 0:
            dp = -exponent
            if dp > max_dp:
                max_dp = dp
    return max_dp

###############################################################################
# 2. Uniform Distractor Generation: Discard old if bounds expand
###############################################################################

def generate_uniform_distractors(
    original_options,
    correct_answer,
    how_many,
    base_decimals,
    max_attempts_factor=1000
):
    """
    Generates 'how_many' distractors as floats, uniformly sampled in [mean-3*sigma, mean+3*sigma].
    
    Expansion pattern:
      - expansions=0 => no change from base
      - expansions=1 => double bounds
      - expansions=2 => double bounds
      - expansions=3 => increment decimal precision by 1
      - expansions=4 => double bounds
      - expansions=5 => double bounds
      - expansions=6 => increment decimal precision by 1
    If we still can't get enough distractors after expansions=6, raise an error.
    
    Implementation details:
      1) If stdev=0 => fallback stdev=1.
      2) If all original options >= 0, skip negative candidates.
      3) Each time we fail to gather how_many distractors in the given attempts, we discard them,
         apply the next expansion step, and try again.
      4) Finally, we format them with the final decimal precision
         (base_decimals + total_decimal_increments).
    """
    mean = statistics.mean(original_options)
    stdev = statistics.pstdev(original_options)
    if stdev == 0.0:
        stdev = 1.0
    
    lower_bound = mean - 3 * stdev
    upper_bound = mean + 3 * stdev
    
    # Decide if negative distractors are allowed
    allow_negative = any(opt < 0 for opt in original_options)
    
    # The expansions pattern
    # expansions=0 => no change
    # expansions=1 => "bounds"
    # expansions=2 => "bounds"
    # expansions=3 => "decimal"
    # expansions=4 => "bounds"
    # expansions=5 => "bounds"
    # expansions=6 => "decimal"
    expansion_actions = [
        None,        # expansions=0 => no change
        "bounds",    # expansions=1 => double bounds
        "bounds",    # expansions=2 => double bounds
        "decimal",   # expansions=3 => +1 decimal
        "bounds",    # expansions=4 => double bounds
        "bounds",    # expansions=5 => double bounds
        "decimal"    # expansions=6 => +1 decimal
    ]
    
    needed = how_many
    max_expansions = len(expansion_actions) - 1  # 6 expansions after the initial attempt
    
    # We'll track how many times we've incremented decimals
    decimal_increments = 0
    
    distractors_floats = None
    
    for expansions_done in range(len(expansion_actions)):
        # expansions_done from 0 to 6 inclusive
        #  - 0 => no change
        #  - 1..6 => see expansion_actions

        if expansions_done > 0:
            # Apply the action from expansion_actions[expansions_done]
            action = expansion_actions[expansions_done]
            if action == "bounds":
                # Double the distance from mean
                span_low = lower_bound - mean
                span_high = upper_bound - mean
                lower_bound = mean + 2 * span_low
                upper_bound = mean + 2 * span_high
            elif action == "decimal":
                # Increase decimal precision by 1
                decimal_increments += 1

        # After possibly adjusting bounds or decimals, attempt generation
        distractors_floats = set()  # discard old distractors each time
        current_decimals = base_decimals + decimal_increments
        
        attempts = 0
        max_attempts = needed * max_attempts_factor
        
        while len(distractors_floats) < needed and attempts < max_attempts:
            candidate = random.uniform(lower_bound, upper_bound)
            attempts += 1
            
            if not allow_negative and candidate < 0:
                # skip negative
                continue
            
            # Round to current_decimals
            candidate_str = f"{candidate:.{current_decimals}f}"
            candidate_val = float(candidate_str)
            
            # Skip if effectively correct
            if abs(candidate_val - correct_answer) < 1e-14:
                continue
            
            distractors_floats.add(candidate_val)
        
        # Check if we succeeded
        if len(distractors_floats) >= needed:
            # success
            final_decimals = current_decimals
            # We can break out of the expansions loop
            break
        else:
            # Not enough
            if expansions_done == max_expansions:
                # i.e. expansions_done=6 => we've exhausted expansions
                raise ValueError(
                    f"Could not generate {needed} unique distractors after {expansions_done} expansions."
                )
            # Otherwise, continue to the next expansions_done iteration

    # Here we have enough distractors
    final_list_floats = list(distractors_floats)[:needed]
    # Format them with final_decimals
    distractors_str_list = [f"{val:.{final_decimals}f}" for val in final_list_floats]
    return distractors_str_list, final_decimals


###############################################################################
# 3. Expanding Each Question
###############################################################################

def expand_numeric_answers_for_question(question_data, N):
    """
    Produces N answers for one question:
      1) The correct answer (formatted at final precision)
      2) (N-1) new distractors from uniform distribution
         with your specified expansion pattern:
           - expansions=1,2 => double bounds
           - expansions=3 => increment decimals
           - expansions=4,5 => double bounds
           - expansions=6 => increment decimals
    """
    correct_answer = question_data['correct_answer']
    original_options = question_data['options']
    M = N - 1
    
    base_decimals = get_max_decimals_in_original(original_options)
    
    # If M <= 0 => means N=0 or N=1 => no distractors needed
    if M <= 0:
        final_decimals = base_decimals
        correct_str = f"{correct_answer:.{final_decimals}f}"
        return [correct_str]
    
    # Otherwise, generate M distractors with pattern expansions
    distractors_list, final_decimals = generate_uniform_distractors(
        original_options=original_options,
        correct_answer=correct_answer,
        how_many=M,
        base_decimals=base_decimals,
        max_attempts_factor=1000
    )
    correct_str = f"{correct_answer:.{final_decimals}f}"
    
    all_answers = [correct_str] + distractors_list
    random.shuffle(all_answers)
    return all_answers

###############################################################################
# 4. Generating the Full Quiz & Writing CSV
###############################################################################

def generate_expanded_quiz_numeric(csv_file_path, N):
    """
    1) Reads numeric questions from CSV.
    2) For each question:
       - Expand to N total answers (1 correct + N-1 distractors)
         via the expansion pattern above.
    3) Returns a list of dicts:
       {
         'question': <str>,
         'correct_answer': <str>,
         'answers': [<str>, ...]
       }
    """
    questions_data = read_numeric_questions_from_csv(csv_file_path)
    expanded_questions = []
    
    for q in questions_data:
        final_answers = expand_numeric_answers_for_question(q, N)
        
        # We can detect final decimals from the first answer (if any),
        # or fall back to base in some edge cases
        dec_count = 0
        if final_answers and '.' in final_answers[0]:
            dec_count = len(final_answers[0].split('.')[1])
        
        correct_answer_formatted = f"{q['correct_answer']:.{dec_count}f}"
        
        expanded_questions.append({
            'question': q['question'],
            'correct_answer': correct_answer_formatted,
            'answers': final_answers
        })
    
    return expanded_questions


def write_expanded_numeric_questions_to_csv(expanded_questions, output_file_path):
    """
    Creates a CSV:
      - 'question' in column 1
      - columns A, B, C, ..., for each numeric answer
      - 'correct_answer' in the last column, containing the label of the correct answer's column.
    """
    import csv
    
    max_answers = max((len(q['answers']) for q in expanded_questions), default=0)
    
    # Generate column labels for up to max_answers
    labels = generate_labels(max_answers)  # e.g. ["A","B","C","D","E","F"] if max_answers=6
    
    # Fieldnames: question, then those labels, then correct_answer
    fieldnames = ["question"] + labels + ["correct_answer"]
    
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in expanded_questions:
            row_data = {}
            row_data["question"] = item["question"]
            
            # We'll assume 'answers' is a list of numeric strings,
            # and 'correct_answer' is the correct numeric string.
            answers = item["answers"]
            correct_str = item["correct_answer"]
            
            # Fill columns for each answer
            for i, ans_val in enumerate(answers):
                row_data[labels[i]] = ans_val
            
            # Identify the correct answer's index
            if correct_str in answers:
                idx_correct = answers.index(correct_str)
                row_data["correct_answer"] = labels[idx_correct]
            else:
                row_data["correct_answer"] = ""
            
            writer.writerow(row_data)


def write_zero_answer_questions_to_csv(zero_answer_data, output_file_path):
    """
    Writes a CSV with just two columns: question, correct_answer
    """
    import csv

    fieldnames = ["question", "correct_answer"]

    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in zero_answer_data:
            writer.writerow({
                "question": item["question"],
                "correct_answer": item["correct_answer"]
            })

def main():
    input_csv = 'Physics Test Questions.csv'
    output_csv = '0v2_alp.csv'
    desired_num_answers = 0  # example
    
    if desired_num_answers == 0:
        # Special path
        expanded_data = generate_zero_answer_questions(input_csv)
        write_zero_answer_questions_to_csv(expanded_data, output_csv)
    else:
        # Normal path
        expanded_data = generate_expanded_quiz_numeric(input_csv, desired_num_answers)
        write_expanded_numeric_questions_to_csv(expanded_data, output_csv)

if __name__ == "__main__":
    main()

