#!/usr/bin/env python3
# filepath: /Users/ibrahimteymurlu/Documents/University of Eindhoven/Master Thesis/Repos/mammoth/scripts/neuron_analyzer.py

import os
import re
import glob
import pandas as pd
import numpy as np
from collections import defaultdict


class NeuronAnalyzer:
    """
    Analyzes neural network logs to track inactive and reviving neurons across tasks and their
    correlation with performance.
    """

    def __init__(self, log_dir):
        """Initialize with directory containing log files."""
        self.log_dir = log_dir
        self.log_files = glob.glob(os.path.join(log_dir, "*.log"))
        self.inactive_neurons = defaultdict(list)
        self.reviving_neurons = defaultdict(list)
        self.accuracies = defaultdict(list)

    def parse_log_files(self):
        """Parse all log files to extract neuron statistics and accuracies."""
        for log_file in self.log_files:
            model_name = os.path.basename(log_file).split("_seed")[0]
            print(
                f"Analyzing model: {model_name} from file {os.path.basename(log_file)}"
            )

            with open(log_file, "r") as f:
                content = f.read()

            # Extract task information
            tasks = re.findall(r"=== End of Task (\d+) Analysis ===", content)
            max_task = int(tasks[-1]) if tasks else 0
            print(f"  Found {max_task+1} tasks")

            # Find all reviving neuron instances across the entire log first
            all_reviving_matches = []

            # First, search with a more specific pattern that checks for neuron counts
            for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
                # More specific pattern that requires the neuron count and IDs
                pattern = f"Reviving Neurons Analysis for {layer} at Task (\\d+):(?:[^R]+?)Number of reviving neurons: (\\d+)(?:[^R]+?)Reviving neurons in {layer}: \\[(.*?)\\](?:[^R]+?)Re-initializing (\\d+) reviving neurons in {layer}"

                try:
                    matches = re.finditer(pattern, content)

                    for match in matches:
                        actual_task = int(match.group(1))
                        reported_count = int(match.group(2))
                        neuron_ids = match.group(3).strip()
                        reinit_count = int(match.group(4))

                        # Verify counts for consistency
                        neuron_list = (
                            [int(x.strip()) for x in neuron_ids.split(",")]
                            if neuron_ids
                            else []
                        )
                        actual_count = len(neuron_list)

                        # Check if counts are consistent
                        if reported_count != reinit_count or (
                            actual_count > 0 and actual_count != reported_count
                        ):
                            print(
                                f"  WARNING: Count mismatch in {layer} at Task {actual_task}: "
                                + f"Reported {reported_count}, Reinitialized {reinit_count}, "
                                + f"Actual IDs count {actual_count}"
                            )

                            # Use the most reliable count (typically the one from neuron IDs)
                            if actual_count > 0:
                                reviving_count = actual_count
                            else:
                                # Fall back to reinitialization count
                                reviving_count = reinit_count
                        else:
                            reviving_count = reinit_count

                        # Skip entries with zero reviving neurons
                        if reviving_count == 0:
                            print(
                                f"  Skipping zero-count reviving entry for {layer} at Task {actual_task}"
                            )
                            continue

                        all_reviving_matches.append(
                            {
                                "task": actual_task,
                                "layer": layer,
                                "reviving_count": reviving_count,
                                "neuron_ids": neuron_ids,
                            }
                        )
                        print(
                            f"  Found {reviving_count} reviving neurons in {layer} at Task {actual_task}"
                        )
                except Exception as e:
                    print(f"  Error processing {layer} reviving neurons: {e}")

            # Add all found reviving neurons to the dictionary
            for record in all_reviving_matches:
                self.reviving_neurons[model_name].append(record)

            # Extract inactive neurons for ALL epochs
            # Find all epoch blocks in the entire log
            epoch_pattern = r"Epoch (\d+):"
            epoch_matches = re.finditer(epoch_pattern, content)

            # Process each epoch in the log file
            for epoch_match in epoch_matches:
                epoch_num = int(epoch_match.group(1))
                epoch_start = epoch_match.start()

                # Find the end of this epoch section
                next_epoch_match = re.search(r"Epoch \d+:", content[epoch_start + 10 :])
                if next_epoch_match:
                    epoch_end = epoch_start + 10 + next_epoch_match.start()
                else:
                    epoch_end = len(content)

                epoch_content = content[epoch_start:epoch_end]

                # Determine which task this epoch belongs to
                # Check for task markers before this epoch
                task_markers = re.findall(r"Task (\d+) - Epoch", content[:epoch_start])
                current_task = int(task_markers[-1]) if task_markers else 0

                # Extract inactive neurons per layer for this epoch
                for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
                    inactive_pattern = (
                        rf"Layer: {layer}\s+Inactive Neurons Count:\s+(\d+)"
                    )
                    inactive_match = re.search(inactive_pattern, epoch_content)

                    if inactive_match:
                        inactive_count = int(inactive_match.group(1))
                        self.inactive_neurons[model_name].append(
                            {
                                "task": current_task,
                                "layer": layer,
                                "inactive_count": inactive_count,
                                "epoch": epoch_num,
                            }
                        )
                        print(
                            f"  Task {current_task}, Epoch {epoch_num}, {layer}: {inactive_count} inactive neurons"
                        )

            # Extract accuracies
            accuracy_pattern = r"Accuracy for (\d+) task\(s\): \s+\[Class-IL\]: ([\d.]+) % \s+\[Task-IL\]: ([\d.]+) %\s+Raw accuracy values: Class-IL \[(.*?)\] \| Task-IL \[(.*?)\]"
            accuracy_matches = re.findall(accuracy_pattern, content)

            for match in accuracy_matches:
                num_tasks, class_il_avg, task_il_avg, class_il_raw, task_il_raw = match

                # Parse raw accuracy values
                class_il_values = [float(x.strip()) for x in class_il_raw.split(",")]
                task_il_values = [float(x.strip()) for x in task_il_raw.split(",")]

                self.accuracies[model_name].append(
                    {
                        "num_tasks": int(num_tasks),
                        "class_il_avg": float(class_il_avg),
                        "task_il_avg": float(task_il_avg),
                        "class_il_raw": class_il_values,
                        "task_il_raw": task_il_values,
                    }
                )

        # Convert to DataFrames for easier analysis
        self.inactive_df = pd.DataFrame(
            [
                {**record, "model": model}
                for model, records in self.inactive_neurons.items()
                for record in records
            ]
        )

        # Create reviving DataFrame excluding neuron_ids from CSV output
        reviving_records = []
        for model, records in self.reviving_neurons.items():
            for record in records:
                # Create a clean record without neuron_ids for the CSV
                clean_record = {
                    "task": record["task"],
                    "layer": record["layer"],
                    "reviving_count": record["reviving_count"],
                    "model": model,
                }
                reviving_records.append(clean_record)
        self.reviving_df = pd.DataFrame(reviving_records)
        print(
            f"Found {len(reviving_records)} reviving neuron records across all models"
        )

        # Process accuracy data
        accuracy_records = []
        for model, acc_list in self.accuracies.items():
            for entry in acc_list:
                num_tasks = entry["num_tasks"]

                # Record each task's individual accuracy
                for task_idx in range(num_tasks):
                    if task_idx < len(entry["class_il_raw"]):
                        accuracy_records.append(
                            {
                                "model": model,
                                "num_tasks_seen": num_tasks,
                                "task": task_idx,
                                "metric_type": "Class-IL",
                                "accuracy": entry["class_il_raw"][task_idx],
                            }
                        )

                    if task_idx < len(entry["task_il_raw"]):
                        accuracy_records.append(
                            {
                                "model": model,
                                "num_tasks_seen": num_tasks,
                                "task": task_idx,
                                "metric_type": "Task-IL",
                                "accuracy": entry["task_il_raw"][task_idx],
                            }
                        )

                # Add average accuracies
                accuracy_records.append(
                    {
                        "model": model,
                        "num_tasks_seen": num_tasks,
                        "task": "avg",
                        "metric_type": "Class-IL",
                        "accuracy": entry["class_il_avg"],
                    }
                )

                accuracy_records.append(
                    {
                        "model": model,
                        "num_tasks_seen": num_tasks,
                        "task": "avg",
                        "metric_type": "Task-IL",
                        "accuracy": entry["task_il_avg"],
                    }
                )

        self.accuracy_df = pd.DataFrame(accuracy_records)

    def save_to_csv(self, output_dir=None):
        """Save extracted data to CSV files."""
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, "analysis")

        os.makedirs(output_dir, exist_ok=True)

        # Save the DataFrames to CSV
        self.inactive_df.to_csv(
            os.path.join(output_dir, "inactive_neurons.csv"), index=False
        )
        self.reviving_df.to_csv(
            os.path.join(output_dir, "reviving_neurons.csv"), index=False
        )
        self.accuracy_df.to_csv(os.path.join(output_dir, "accuracies.csv"), index=False)

        print(f"CSV files saved to {output_dir}")
        return output_dir

    def debug_log_file(self, log_file=None):
        """Debug a specific log file to identify reviving neuron issues."""
        if log_file is None and self.log_files:
            log_file = self.log_files[0]

        print(f"Debugging log file: {os.path.basename(log_file)}")

        with open(log_file, "r") as f:
            content = f.read()

        # Find all instances of inactive neurons by epoch
        epoch_pattern = r"Epoch (\d+):"
        epoch_matches = re.finditer(epoch_pattern, content)

        for epoch_match in epoch_matches:
            epoch_num = int(epoch_match.group(1))
            epoch_start = epoch_match.start()

            # Determine which task this epoch belongs to
            task_markers = re.findall(r"Task (\d+) - Epoch", content[:epoch_start])
            current_task = int(task_markers[-1]) if task_markers else 0

            # Find the end of this epoch section
            next_epoch_match = re.search(r"Epoch \d+:", content[epoch_start + 10 :])
            if next_epoch_match:
                epoch_end = epoch_start + 10 + next_epoch_match.start()
            else:
                epoch_end = len(content)

            epoch_content = content[epoch_start:epoch_end]

            print(f"\nTask {current_task}, Epoch {epoch_num}:")
            for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
                inactive_pattern = rf"Layer: {layer}\s+Inactive Neurons Count:\s+(\d+)"
                inactive_match = re.search(inactive_pattern, epoch_content)

                if inactive_match:
                    inactive_count = int(inactive_match.group(1))
                    print(f"  {layer}: {inactive_count} inactive neurons")

        # Find all instances of reviving neurons
        for layer in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            simple_pattern = f"Reviving Neurons Analysis for {layer} at Task (\\d+):"
            simple_matches = re.finditer(simple_pattern, content)

            for match in re.finditer(simple_pattern, content):
                task = int(match.group(1))
                start_pos = match.start()

                # Extract 500 characters after the match to see full context
                context = content[start_pos : start_pos + 500]

                # Look for all number mentions in the context
                count_pattern = r"Number of reviving neurons: (\d+)"
                reinit_pattern = f"Re-initializing (\\d+) reviving neurons in {layer}"

                count_match = re.search(count_pattern, context)
                reinit_match = re.search(reinit_pattern, context)

                count = int(count_match.group(1)) if count_match else "Not found"
                reinit = int(reinit_match.group(1)) if reinit_match else "Not found"

                print(f"\nTask {task}, {layer}:")
                print(f"  Number of reviving neurons: {count}")
                print(f"  Re-initializing count: {reinit}")
                print(f"  Context snippet: {context[:100]}...")


def main():
    # Base directory containing multiple log directories
    base_dir = "/Users/ibrahimteymurlu/Documents/University of Eindhoven/Master Thesis/Repos/mammoth/task_accuracies/elahe_feedback"

    # Find all log directories (assuming each directory contains .log files)
    log_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        # Check if it's a directory and contains .log files
        if os.path.isdir(item_path) and any(
            file.endswith(".log") for file in os.listdir(item_path)
        ):
            log_dirs.append(item_path)

    if not log_dirs:
        print("No log directories found. Please check the base directory path.")
        return

    print(f"Found {len(log_dirs)} log directories to analyze.")

    # Process each log directory
    for log_dir in log_dirs:
        print(f"\nProcessing log directory: {log_dir}")
        analyzer = NeuronAnalyzer(log_dir)
        analyzer.parse_log_files()

        # Save CSVs
        csv_dir = analyzer.save_to_csv()
        print(f"CSV files saved to {csv_dir}")

        # Uncomment to debug first log file in each directory
        # if analyzer.log_files:
        #     analyzer.debug_log_file(analyzer.log_files[0])

    print("\nAnalysis completed for all log directories.")


if __name__ == "__main__":
    main()
