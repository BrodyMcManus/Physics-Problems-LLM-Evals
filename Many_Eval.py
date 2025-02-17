import os
import statistics

# Set the log directory (this will tell Inspect to write logs here).
os.environ["INSPECT_LOG_DIR"] = "./experiment_logs"
os.makedirs("./experiment_logs", exist_ok=True)

# Import the eval() function and your task function.
from inspect_ai import eval
from MQC_noreasoning_eval import custom_mc_no_cot_eval

# Import your dataset modifier function.
# This function should take the original CSV path and a run number,
# produce a new CSV (e.g. by modifying some content), and return the new CSV file path.
from QuestionGenerator.qgenv2_alpha import modq

# Number of evaluations to run
NUM_EVALS = 2
eval_logs = []

# Path to your original dataset CSV.
original_csv = "Physics Test Questions Short.csv"
distractors = 10

print("Starting batch evaluation runs...")

# Run evaluations in a loop.
for i in range(NUM_EVALS):
    # Produce a new modified CSV file.
    new_csv = modq(original_csv, distractors)
    print(f"Run {i+1}: using CSV file {new_csv}")

    # Instantiate the task with the new CSV.
    task_instance = custom_mc_no_cot_eval(new_csv)
    
    # Run the evaluation (adjust the model name as needed).
    # The eval() call returns an EvalLog (or a list of EvalLogs).
    log_obj = eval(task_instance, model="google/gemini-1.5-flash", log_dir="./experiment_logs")
    
    # If eval() returns a list, extend our list; otherwise, append the single log.
    if isinstance(log_obj, list):
        eval_logs.extend(log_obj)
    else:
        eval_logs.append(log_obj)
    
    print(f"Completed evaluation run {i+1}/{NUM_EVALS}\n")

# Aggregate evaluation statistics.
# The following assumes there is only one scorer
accuracies = []
for i, log_obj in enumerate(eval_logs):
    print(f"Log {i} status: {log_obj.status}")
    if log_obj.status == "success":
        scores = log_obj.results.scores
        if scores:
            # For example, assume there's only one scorer in the list.
            # If you have multiple scorers, you can iterate or filter by name.
            scorer_metrics = scores[0].metrics
            # 'accuracy' is an EvalMetric object with a .value attribute
            accuracy_metric = scorer_metrics.get("accuracy")
            if accuracy_metric is not None:
                acc_val = accuracy_metric.value
                accuracies.append(acc_val)
                print(f"Log {i} accuracy: {acc_val}")
            else:
                print(f"Log {i} has no accuracy metric.")
        else:
            print(f"Log {i} has no scores in results.")
    else:
        print(f"Log {i} is not success. Status = {log_obj.status}")

# Aggregate
if accuracies:
    avg_accuracy = statistics.mean(accuracies)
    std_accuracy = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    print(f"\nAverage Accuracy: {avg_accuracy}")
    print(f"Standard Deviation: {std_accuracy}")
else:
    print("\nNo accuracy metrics found in successful evaluations.")

# Write aggregated statistics to a file.
with open("aggregated_eval_stats.txt", "w") as f:
    f.write(f"Number of Evaluations: {NUM_EVALS}\n")
    if accuracies:
        f.write(f"Average Accuracy: {avg_accuracy}\n")
        f.write(f"Standard Deviation: {std_accuracy}\n")
    else:
        f.write("No successful evaluations found.\n")

print("Batch evaluation complete. Aggregated statistics written to aggregated_eval_stats.txt")
