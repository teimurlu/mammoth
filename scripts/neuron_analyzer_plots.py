import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob

# Define the base seeds directory path
seeds_base_path = Path(
    "/Users/ibrahimteymurlu/Documents/University of Eindhoven/Master Thesis/Repos/mammoth/task_accuracies/elahe_feedback/"
)

# Get all subdirectories in the seeds folder (each representing a different seed experiment)
seed_folders = [
    f for f in seeds_base_path.glob("*") if f.is_dir() and "analysis" not in f.name
]


# Function to process each seed folder
def process_seed_folder(base_path):
    print(f"Processing folder: {base_path}")

    # Define paths to CSV files
    analysis_path = base_path / "analysis"
    if not analysis_path.exists():
        print(f"No analysis folder found in {base_path}. Skipping...")
        return

    accuracy_file = analysis_path / "accuracies.csv"
    inactive_file = analysis_path / "inactive_neurons.csv"
    # reviving_file = analysis_path / "reviving_neurons.csv"

    # Check if required files exist
    if not accuracy_file.exists() or not inactive_file.exists():
        print(f"Required CSV files not found in {analysis_path}. Skipping...")
        return

    # Create output directory for plots
    output_dir = analysis_path / "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data
    accuracies = pd.read_csv(accuracy_file)
    inactive_neurons = pd.read_csv(inactive_file)
    # reviving_neurons = pd.read_csv(reviving_file)

    # Function to extract method components
    def parse_model_name(model_name):
        if model_name.startswith("01_baseline"):
            return {
                "activation": "relu",
                "threshold": "0.0",
                "strategy": "baseline",
                "optimizer": "sgd",
            }

        parts = model_name.split("_")
        if len(parts) < 3:
            return {
                "activation": "unknown",
                "threshold": "unknown",
                "strategy": "unknown",
                "optimizer": "unknown",
            }

        activation = parts[0]
        threshold = parts[1]
        strategy = parts[2]
        optimizer = parts[3] if len(parts) > 3 else "unknown"

        return {
            "activation": activation,
            "threshold": threshold,
            "strategy": strategy,
            "optimizer": optimizer,
        }

    # Filter accuracies for Task-IL and average metrics
    task_il_acc = accuracies[accuracies["metric_type"] == "Task-IL"]
    avg_acc = task_il_acc[task_il_acc["task"] == "avg"]

    class_il_acc = accuracies[accuracies["metric_type"] == "Class-IL"]
    avg_class_il_acc = class_il_acc[class_il_acc["task"] == "avg"]

    # 1. Create a comprehensive accuracy comparison table
    def create_task_il_table():
        # Pivot table for all task performances
        task_table = (
            task_il_acc[task_il_acc["task"] != "avg"]
            .pivot_table(
                index=["model"],
                columns=["num_tasks_seen"],
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
        )

        # Add average across all tasks
        avg_data = avg_acc.groupby("model")["accuracy"].mean().reset_index()
        task_table = pd.merge(task_table, avg_data, on="model", how="left")
        task_table.rename(columns={"accuracy": "Average"}, inplace=True)

        # Format table for better readability
        task_table = task_table.round(2)

        return task_table

    def create_class_il_table():
        # Pivot table for all task performances
        class_il_table = (
            class_il_acc[class_il_acc["task"] != "avg"]
            .pivot_table(
                index=["model"],
                columns=["num_tasks_seen"],
                values="accuracy",
                aggfunc="mean",
            )
            .reset_index()
        )

        # Add average across all tasks
        avg_data = avg_class_il_acc.groupby("model")["accuracy"].mean().reset_index()
        class_il_table = pd.merge(class_il_table, avg_data, on="model", how="left")
        class_il_table.rename(columns={"accuracy": "Average"}, inplace=True)

        # Format table for better readability
        class_il_table = class_il_table.round(2)

        return class_il_table

    def plot_accuracy_trends():
        plt.figure(figsize=(14, 8))

        # Final accuracy comparison
        final_acc = avg_acc[avg_acc["num_tasks_seen"] == 5]

        # Create a custom color palette that highlights baseline
        models = final_acc.sort_values("accuracy", ascending=False)["model"].tolist()
        colors = ["red" if "baseline" in model else "steelblue" for model in models]

        plt.figure(figsize=(16, 8))
        sns.barplot(
            x="model",
            y="accuracy",
            data=final_acc,
            palette=colors,
            order=models,
        )
        plt.title("Final Task-IL Accuracy After All Tasks (Task 5)")
        plt.xlabel("Model Configuration")
        plt.ylabel("Task-IL Accuracy (%)")
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "final_accuracy_comparison.png", dpi=300)

        # Class-IL plot with the same highlighting approach
        plt.figure(figsize=(16, 8))
        final_class_il_acc = avg_class_il_acc[avg_class_il_acc["num_tasks_seen"] == 5]

        # Create custom color palette for class-IL plot
        class_il_models = final_class_il_acc.sort_values("accuracy", ascending=False)[
            "model"
        ].tolist()
        class_il_colors = [
            "red" if "baseline" in model else "purple" for model in class_il_models
        ]

        sns.barplot(
            x="model",
            y="accuracy",
            data=final_class_il_acc,
            palette=class_il_colors,
            order=class_il_models,
        )
        plt.title("Final Class-IL Accuracy After All Tasks (Task 5)")
        plt.xlabel("Model Configuration")
        plt.ylabel("Class-IL Accuracy (%)")
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "final_class_il_accuracy_comparison.png", dpi=300)

        # Add a legend for better clarity
        for fig_num in [2, 3]:  # Access the last two created figures
            plt.figure(fig_num)
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="red", label="Baseline"),
                Patch(
                    facecolor="steelblue" if fig_num == 2 else "purple",
                    label="Other Models",
                ),
            ]
            plt.legend(handles=legend_elements, loc="best")
            plt.tight_layout()
            plt.savefig(
                output_dir
                / f"final_{'task' if fig_num == 2 else 'class'}_il_accuracy_comparison.png",
                dpi=300,
            )

        # NEW CODE: Add line charts showing accuracy progression across tasks

        # 1. Task-IL progression line chart
        plt.figure(figsize=(14, 8))

        # Prepare data - we want to show avg accuracy across tasks for each model
        task_progression = avg_acc.pivot(
            index="num_tasks_seen", columns="model", values="accuracy"
        )

        # Plot with distinct color for baseline
        for column in task_progression.columns:
            if "baseline" in column:
                plt.plot(
                    task_progression.index,
                    task_progression[column],
                    marker="o",
                    linewidth=3,
                    markersize=10,
                    color="red",
                    label=column,
                )
            else:
                plt.plot(
                    task_progression.index,
                    task_progression[column],
                    marker=".",
                    linewidth=1.5,
                    alpha=0.7,
                    label=column,
                )

        plt.title("Task-IL Accuracy Progression Across Tasks")
        plt.xlabel("Number of Tasks Seen")
        plt.ylabel("Task-IL Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.xticks(task_progression.index)

        # Add legend with smaller font size placed outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        plt.savefig(
            output_dir / "task_il_progression.png", dpi=300, bbox_inches="tight"
        )

        # 2. Class-IL progression line chart
        plt.figure(figsize=(14, 8))

        # Prepare data for Class-IL
        class_il_progression = avg_class_il_acc.pivot(
            index="num_tasks_seen", columns="model", values="accuracy"
        )

        # Plot with distinct color for baseline
        for column in class_il_progression.columns:
            if "baseline" in column:
                plt.plot(
                    class_il_progression.index,
                    class_il_progression[column],
                    marker="o",
                    linewidth=3,
                    markersize=10,
                    color="red",
                    label=column,
                )
            else:
                plt.plot(
                    class_il_progression.index,
                    class_il_progression[column],
                    marker=".",
                    linewidth=1.5,
                    alpha=0.7,
                    label=column,
                )

        plt.title("Class-IL Accuracy Progression Across Tasks")
        plt.xlabel("Number of Tasks Seen")
        plt.ylabel("Class-IL Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.xticks(class_il_progression.index)

        # Add legend with smaller font size placed outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        plt.savefig(
            output_dir / "class_il_progression.png", dpi=300, bbox_inches="tight"
        )

    def analyze_inactive_neurons():
        # Focus on the final task and last epoch
        final_inactive = inactive_neurons[
            (inactive_neurons["task"] == 5)
            & (inactive_neurons["epoch"] == inactive_neurons["epoch"].max())
        ]

        # Merge inactive neuron counts with Task-IL accuracy data
        accuracy_task5 = avg_acc[avg_acc["num_tasks_seen"] == 5]

        # Make sure we're merging correctly and preserving the activation column
        merged_data = pd.merge(
            final_inactive[final_inactive["layer"] == "layer4"],
            accuracy_task5,
            on="model",
            how="inner",
        )

        # Check if activation column exists in merged_data
        if (
            "activation_x" in merged_data.columns
            and "activation_y" in merged_data.columns
        ):
            # If we have duplicate columns, use one and rename it
            merged_data["activation"] = merged_data["activation_x"]
            merged_data.drop(["activation_x", "activation_y"], axis=1, inplace=True)
        elif "activation" not in merged_data.columns:
            # If activation is missing entirely, add it back from the model names
            merged_data["activation"] = merged_data["model"].apply(
                lambda x: parse_model_name(x)["activation"]
            )

        # Same for optimizer
        if (
            "optimizer_x" in merged_data.columns
            and "optimizer_y" in merged_data.columns
        ):
            merged_data["optimizer"] = merged_data["optimizer_x"]
            merged_data.drop(["optimizer_x", "optimizer_y"], axis=1, inplace=True)
        elif "optimizer" not in merged_data.columns:
            merged_data["optimizer"] = merged_data["model"].apply(
                lambda x: parse_model_name(x)["optimizer"]
            )

        # Plot layer4 inactive neuron count vs. Task-IL accuracy
        plt.figure(figsize=(12, 8))

        # Ensure the columns exist before using them
        if "activation" in merged_data.columns and "optimizer" in merged_data.columns:
            sns.scatterplot(
                data=merged_data,
                x="inactive_count",
                y="accuracy",
                hue="activation",
                style="optimizer",
                s=100,
                alpha=0.7,
            )
        else:
            # Fallback if columns are missing
            sns.scatterplot(
                data=merged_data,
                x="inactive_count",
                y="accuracy",
                s=100,
                alpha=0.7,
            )

        # Add text labels for points
        for _, row in merged_data.iterrows():
            plt.text(
                row["inactive_count"] + 1,
                row["accuracy"],
                row["model"],
                fontsize=8,
                alpha=0.8,
            )

        plt.title(
            "Relationship Between Inactive Neurons in Layer 4 and Final Task-IL Accuracy"
        )
        plt.xlabel("Number of Inactive Neurons")
        plt.ylabel("Task-IL Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "inactive_neurons_vs_accuracy.png", dpi=300)

        # NEW CODE: Create the same plot for Class-IL accuracy
        # Merge inactive neuron counts with Class-IL accuracy data
        class_il_accuracy_task5 = avg_class_il_acc[
            avg_class_il_acc["num_tasks_seen"] == 5
        ]

        merged_class_il_data = pd.merge(
            final_inactive[final_inactive["layer"] == "layer4"],
            class_il_accuracy_task5,
            on="model",
            how="inner",
        )

        # Handle columns just like in the Task-IL case
        if (
            "activation_x" in merged_class_il_data.columns
            and "activation_y" in merged_class_il_data.columns
        ):
            merged_class_il_data["activation"] = merged_class_il_data["activation_x"]
            merged_class_il_data.drop(
                ["activation_x", "activation_y"], axis=1, inplace=True
            )
        elif "activation" not in merged_class_il_data.columns:
            merged_class_il_data["activation"] = merged_class_il_data["model"].apply(
                lambda x: parse_model_name(x)["activation"]
            )

        if (
            "optimizer_x" in merged_class_il_data.columns
            and "optimizer_y" in merged_class_il_data.columns
        ):
            merged_class_il_data["optimizer"] = merged_class_il_data["optimizer_x"]
            merged_class_il_data.drop(
                ["optimizer_x", "optimizer_y"], axis=1, inplace=True
            )
        elif "optimizer" not in merged_class_il_data.columns:
            merged_class_il_data["optimizer"] = merged_class_il_data["model"].apply(
                lambda x: parse_model_name(x)["optimizer"]
            )

        # Create Class-IL plot
        plt.figure(figsize=(12, 8))

        if (
            "activation" in merged_class_il_data.columns
            and "optimizer" in merged_class_il_data.columns
        ):
            sns.scatterplot(
                data=merged_class_il_data,
                x="inactive_count",
                y="accuracy",
                hue="activation",
                style="optimizer",
                s=100,
                alpha=0.7,
            )
        else:
            sns.scatterplot(
                data=merged_class_il_data,
                x="inactive_count",
                y="accuracy",
                s=100,
                alpha=0.7,
            )

        # Add text labels for points
        for _, row in merged_class_il_data.iterrows():
            plt.text(
                row["inactive_count"] + 1,
                row["accuracy"],
                row["model"],
                fontsize=8,
                alpha=0.8,
            )

        plt.title(
            "Relationship Between Inactive Neurons in Layer 4 and Final Class-IL Accuracy"
        )
        plt.xlabel("Number of Inactive Neurons")
        plt.ylabel("Class-IL Accuracy (%)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "inactive_neurons_vs_class_il_accuracy.png", dpi=300)

        # Create table showing inactive neuron progression
        inactive_progression = inactive_neurons[
            (inactive_neurons["layer"] == "layer4")
            & (inactive_neurons["epoch"] == 2)  # Last epoch
        ].pivot_table(
            index="model", columns="task", values="inactive_count", aggfunc="mean"
        )

        return inactive_progression

    # Execute the analysis functions
    task_il_table = create_task_il_table()
    class_il_table = create_class_il_table()
    plot_accuracy_trends()
    inactive_progression = analyze_inactive_neurons()

    # Save tables to CSV
    task_il_table.to_csv(output_dir / "task_il_accuracy_table.csv", index=False)
    class_il_table.to_csv(output_dir / "class_il_accuracy_table.csv", index=False)
    inactive_progression.to_csv(output_dir / "inactive_neurons_progression.csv")

    plt.close("all")  # Close all figure windows to free memory

    # Return summary info
    return base_path.name, len(task_il_table), len(class_il_table)


# Loop through all seed folders and process each one
results = []
for seed_folder in seed_folders:
    try:
        result = process_seed_folder(seed_folder)
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error processing folder {seed_folder}: {str(e)}")

# Print summary of all processed folders
print("\n===== Summary of Analysis =====")
print(f"Total folders processed: {len(results)}")
print("\nDetails:")
for folder_name, task_models, class_models in results:
    print(
        f"- {folder_name}: {task_models} models analyzed (Task-IL), {class_models} models (Class-IL)"
    )

print("\nAnalysis complete!")
