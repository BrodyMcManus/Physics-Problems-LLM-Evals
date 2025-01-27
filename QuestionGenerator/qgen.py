import csv
import random
import statistics
import math
from decimal import Decimal

def read_numeric_questions_from_csv(csv_file_path):
    """
    Reads a CSV where each row has:
       Question, OptionA, OptionB, OptionC, OptionD, CorrectAnswer
    with OptionA/B/C/D as numeric strings, and CorrectAnswer as one
    of "A", "B", "C", or "D". Returns a list of dicts.
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
                    f"CorrectAnswer must be one of A, B, C, or D. Got {correct_letter}"
                )
            
            questions_data.append({
                'question': row['Question'],
                'options': [option_a, option_b, option_c, option_d],
                'correct_answer': correct_ans
            })
    return questions_data

def get_max_decimals_in_original(options):
    """
    Returns the max number of decimal places that appear among the given floats.
    E.g., [3.14, 2.5, 5.0001] -> 4
    """
    max_dp = 0
    for val in options:
        dec = Decimal(str(val)).normalize()
        # dec.as_tuple().exponent is negative if there's a decimal part
        # e.g. "3.1415" -> exponent=-4 => 4 decimals
        exponent = dec.as_tuple().exponent
        if exponent < 0:
            dp = -exponent
            if dp > max_dp:
                max_dp = dp
    return max_dp

def get_decimal_places_for_question(original_options, N):
    """
    Computes how many decimal places we will use for this question.
    1) The max decimal places in the original 4 options.
    2) The log10(N) + 2 approach to ensure we can handle large N.
    Returns max of these two values.
    """
    # 1) decimal places from the original data
    original_dp = get_max_decimals_in_original(original_options)
    
    # 2) decimal places from the desired N
    if N <= 1:
        n_dp = 0
    else:
        n_dp = int(math.floor(math.log10(N))) + 3
    
    return max(original_dp, n_dp)

def generate_uniform_distractors(
    original_options,
    correct_answer,
    how_many,
    max_decimals,
    max_attempts_factor=1000
):
    """
    Generates 'how_many' float distractors by uniformly sampling from
    [mean - 3*stdev,  mean + 3*stdev].
    
    - If all original options are identical or stdev=0, fall back to stdev=1.0.
    - We skip negative values (if you need that domain).
    - We skip exact correct values.
    - We format the final distractors to 'max_decimals' decimals.
    - 'max_attempts_factor' helps prevent infinite loops.
    """
    mean = statistics.mean(original_options)
    
    # If stdev is 0 => fallback
    if len(set(original_options)) == 1:
        stdev = 1.0
    else:
        stdev = statistics.stdev(original_options)
        if stdev == 0.0:
            stdev = 1.0
    
    lower_bound = mean - 3 * stdev
    upper_bound = mean + 3 * stdev
    # Decide if we allow negative distractors or not
    allow_negative = any(opt < 0 for opt in original_options)

    distractors = set()
    max_attempts = how_many * max_attempts_factor
    attempts = 0
    
    while len(distractors) < how_many and attempts < max_attempts:
        candidate = random.uniform(lower_bound, upper_bound)
        attempts += 1
        
        # If all originals are >= 0, we skip negatives
        if (not allow_negative) and (candidate < 0):
            continue
        
        # Format to desired decimal places
        candidate_str = f"{candidate:.{max_decimals}f}"
        candidate_val = float(candidate_str)
        
        # Skip if it's effectively the correct answer after rounding
        if abs(candidate_val - correct_answer) < 1e-14:
            continue
        
        distractors.add(candidate_val)
    
    # Convert them all to the string form with final formatting
    final_list = [f"{val:.{max_decimals}f}" for val in distractors]
    
    # If we can't get enough after max_attempts
    if len(final_list) < how_many:
        raise ValueError(
            f"Could not generate {how_many} unique distractors. "
            f"Got {len(final_list)} after {attempts} attempts. Possibly too many constraints."
            "Try increasing max_attempts_factor or adjusting the logic."
            f"The original options were {original_options} with standard deviation {stdev}"
        )
    
    # Return exactly how_many
    return final_list[:how_many]

def expand_numeric_answers_for_question(question_data, N):
    """
    Produces N answers for a question:
      - 1 correct answer
      - (N-1) newly generated distractors (floats, up to some decimal precision).
    
    We do NOT rely on "all options integer." We always use floats, with decimal 
    precision determined by N.
    """
    correct_answer = question_data['correct_answer']
    original_options = question_data['options']  # includes correct
    M = N - 1
    
    # Decide how many decimal places
    max_decimals = get_decimal_places_for_question(original_options, N)
    
    # Generate M distractors from uniform distribution
    new_distractors = generate_uniform_distractors(
        original_options=original_options,
        correct_answer=correct_answer,
        how_many=M,
        max_decimals=max_decimals
    )
    
    # Combine correct answer + distractors
    # We'll format the correct answer to the same decimal places
    correct_formatted = f"{correct_answer:.{max_decimals}f}"
    all_answers = [correct_formatted] + new_distractors
    random.shuffle(all_answers)
    
    return all_answers

def generate_expanded_quiz_numeric(csv_file_path, N):
    """
    1) Reads numeric questions from CSV.
    2) For each question, produce N answers (1 correct, N-1 distractors).
    3) Returns a list of dicts with the final data.
    """
    questions_data = read_numeric_questions_from_csv(csv_file_path)
    expanded_questions = []
    
    for q in questions_data:
        expanded_answers = expand_numeric_answers_for_question(q, N)
        
        # The correct answer in 'q' is a float; format it to the same decimal 
        # as the expanded answers used. 
        # Actually, we already did that in expand_numeric_answers_for_question, 
        # so let's just identify it from that list or store it separately.
        
        # We can store the correct one in the same format as well:
        max_decimals = get_decimal_places_for_question(q['options'], N)
        correct_answer_str = f"{q['correct_answer']:.{max_decimals}f}"
        
        expanded_questions.append({
            'question': q['question'],
            'correct_answer': correct_answer_str,
            'answers': expanded_answers
        })
    
    return expanded_questions

def write_expanded_numeric_questions_to_csv(expanded_questions, output_file_path):
    """
    Writes the expanded questions to a CSV with columns:
      question, answer_1, ..., answer_M, correct_answer (in the last column).
    """
    import csv

    max_answers = max(len(item['answers']) for item in expanded_questions)
    
    # We want "question", then answer_1..answer_k, then correct_answer
    fieldnames = (
        ["question"] +
        [f"answer_{i+1}" for i in range(max_answers)] +
        ["correct_answer"]
    )
    
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in expanded_questions:
            row_data = {"question": item["question"]}
            
            # Fill answers
            for i, ans in enumerate(item["answers"]):
                row_data[f"answer_{i+1}"] = ans
            
            # Then correct in last column
            row_data["correct_answer"] = item["correct_answer"]
            writer.writerow(row_data)

def main():
    input_csv = 'Physics Test Questions.csv'
    output_csv = '20_expanded_questions.csv'
    desired_num_answers = 20  # for example
    
    expanded_data = generate_expanded_quiz_numeric(input_csv, desired_num_answers)
    write_expanded_numeric_questions_to_csv(expanded_data, output_csv)

if __name__ == "__main__":
    main()



