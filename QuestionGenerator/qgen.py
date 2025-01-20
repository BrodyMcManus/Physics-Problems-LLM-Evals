import csv

def read_questions_from_csv(csv_file_path):
    questions_data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # row should have keys: "Question", "OptionA", "OptionB", "OptionC", "OptionD", "CorrectAnswer"
            question_text = row['Question']
            options = [
                row['OptionA'],
                row['OptionB'],
                row['OptionC'],
                row['OptionD']
            ]
            correct_answer = row['CorrectAnswer']
            
            # Store data in a structured format, e.g. a dict
            question_entry = {
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer
            }
            questions_data.append(question_entry)
    return questions_data

