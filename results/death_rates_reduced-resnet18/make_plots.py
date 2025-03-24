import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from pathlib import Path

# Define the base path for the death rates directory
base_path = Path(
    "/Users/ibrahimteymurlu/Documents/University of Eindhoven/Master Thesis/Repos/mammoth/results/death_rates_resnet18"
)

# Initialize lists to store dataframes
dfs = []

# Death rates to process
death_rates = ["0.1", "0.2", "0.3"]
optimizers = ["adam", "sgd"]
lr_values = {"adam": "0.001", "sgd": "0.1"}

# Loop through each death rate folder and read the overlap files
for rate in death_rates:
    rate_dir = base_path / f"death_rate-{rate}"

    # Check both optimizers
    for opt in optimizers:
        file_path = rate_dir / f"dead_neuron_overlap_opt_{opt}_lr_{lr_values[opt]}.csv"

        if file_path.exists():
            # Read the CSV
            df = pd.read_csv(file_path)

            # Add death rate and optimizer columns
            df["death_rate"] = float(rate)
            df["optimizer"] = opt

            # Append to our list
            dfs.append(df)

# Combine all dataframes
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter for layer4 data and create unique model labels
    layer4_data = combined_df[combined_df["layer"] == "layer4"].copy()
    layer4_data["model"] = layer4_data.apply(
        lambda row: f"{row['optimizer'].capitalize()} ($\\tau$={row['death_rate']})",
        axis=1,
    )

    # Calculate average overlap ratio per task, model
    avg_overlap = (
        layer4_data.groupby(["task", "model", "death_rate", "optimizer"])[
            "overlap_ratio"
        ]
        .mean()
        .reset_index()
    )

    # Create the figure
    plt.figure(figsize=(12, 8))

    # Define custom colors - one set for each optimizer
    colors = ["#d0bfff", "#b2f2bb", "#a5d8ff", "#ffd8a8", "#ffec99", "#ffc9c9"]

    # Create the line plot
    ax = sns.lineplot(
        data=avg_overlap,
        x="task",
        y="overlap_ratio",
        hue="model",
        palette=colors,
        marker="o",
        markersize=8,
        linewidth=2,
    )

    # Customize the plot
    plt.title(
        "Overlap Ratio over Tasks for Different Death Rates and Optimizers",
        fontsize=20,
        pad=20,
    )
    plt.xlabel("Task", fontsize=16)
    plt.ylabel("Overlap Ratio", fontsize=16)

    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Remove the default legend
    ax.get_legend().remove()

    # Create a better positioned legend with consistent formatting
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=14,
    )

    # Customize grid
    plt.grid(True, linestyle="--", alpha=0.3)

    # Set y-axis limits
    plt.ylim(0, 1.1)

    # Set x-axis ticks to show all tasks
    plt.xticks(range(5))

    # Add padding to ensure the plot and legend fit well
    plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.95])

    # Save and show the plot
    plt.savefig(
        base_path / "overlap_ratio_by_death_rate_optimizer.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
else:
    print("No data files were found.")


# For the second plot, we'll analyze the composition of dead neurons
timeline_data = []

# Loop through each death rate folder and read the timeline files
for rate in death_rates:
    rate_dir = base_path / f"death_rate-{rate}"

    # Check both optimizers
    for opt in optimizers:
        file_path = (
            rate_dir / f"layer4_death_timeline_opt_{opt}_lr_{lr_values[opt]}.csv"
        )

        if file_path.exists():
            # Read the CSV
            df = pd.read_csv(file_path)

            # Add death rate and optimizer columns
            df["death_rate"] = float(rate)
            df["optimizer"] = opt

            # Append to our list
            timeline_data.append(df)

# Combine all timeline dataframes
if timeline_data:
    combined_timeline = pd.concat(timeline_data, ignore_index=True)

    # Create model labels
    combined_timeline["model"] = combined_timeline.apply(
        lambda row: f"{row['optimizer'].capitalize()} ($\\tau$={row['death_rate']})",
        axis=1,
    )

    # Selected models to analyze
    selected_models = [
        "Adam ($\\tau$=0.1)",
        "Adam ($\\tau$=0.2)",
        "Adam ($\\tau$=0.3)",
        "Sgd ($\\tau$=0.1)",
        "Sgd ($\\tau$=0.2)",
        "Sgd ($\\tau$=0.3)",
    ]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Custom colors - matches your existing plotting style
    new_color = "#eebefa"  # for newly dead neurons
    persistent_color = "#ffc9c9"  # for persistently dead neurons

    # Process each model
    for i, model in enumerate(selected_models):
        if i < len(axes):
            ax = axes[i]

            # Filter data for this model
            model_data = combined_timeline[combined_timeline["model"] == model]

            # Get unique tasks in timeline data
            tasks = sorted(model_data["death_task"].unique())
            max_task = max(tasks + [4])  # Ensure we have at least 5 tasks

            # Initialize arrays to store counts
            historical_dead = np.zeros(max_task + 1)
            intersection = np.zeros(max_task + 1)

            # For each task, calculate historical dead and intersection
            for task in range(1, max_task + 1):
                # Neurons that died in tasks before the current task
                previous_dead = model_data[model_data["death_task"] < task][
                    "neuron_id"
                ].unique()
                historical_dead[task] = len(previous_dead)

                # Neurons that died in the current task
                current_dead = model_data[model_data["death_task"] == task][
                    "neuron_id"
                ].unique()

                # Total dead neurons up to and including this task
                total_dead = np.concatenate([previous_dead, current_dead])
                historical_dead[task] = len(np.unique(total_dead))

                # Intersection (persistently dead neurons) - these are neurons that died before
                intersection[task] = len(previous_dead)

            # Calculate new dead neurons (historical_dead - intersection)
            new_dead = historical_dead - intersection

            # Create bar plots
            x = range(1, max_task + 1)  # Skip task 0

            # Create stacked bar chart
            ax.bar(
                x, new_dead[1:], label="Newly Dead Neurons", color=new_color, alpha=0.7
            )

            ax.bar(
                x,
                intersection[1:],
                label="Persistently Dead Neurons",
                color=persistent_color,
                bottom=new_dead[1:],
            )

            ax.set_title(model, fontsize=14)
            ax.set_xlabel("Task")
            ax.set_ylabel("Number of Dead Neurons")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_xticks(range(1, max_task + 1))

    # Add a common legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=new_color, alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color=persistent_color),
    ]
    labels = ["Newly Dead Neurons", "Persistently Dead Neurons"]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=14,
    )

    plt.suptitle(
        "Composition of Dead Neurons Across Tasks by Death Rate and Optimizer",
        fontsize=20,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save and show the plot
    plt.savefig(base_path / "neuron_death_composition_by_model.png", dpi=300)
    # plt.show()

    # Optional: Create an alternative visualization comparing all models directly
    plt.figure(figsize=(14, 10))

    # Create a figure with data for all models in a single plot
    x_positions = np.arange(5)  # 5 tasks
    width = 0.12  # Width of each bar
    offsets = np.linspace(-0.3, 0.3, len(selected_models))

    for i, model in enumerate(selected_models):
        # Filter data for this model
        model_data = combined_timeline[combined_timeline["model"] == model]

        # Prepare the historical dead and intersection arrays
        historical_dead = np.zeros(5)
        intersection = np.zeros(5)

        # Calculate for each task
        for task in range(5):
            # For task t, find neurons that died in tasks 0 to t-1
            if task > 0:
                previous_dead = model_data[model_data["death_task"] < task][
                    "neuron_id"
                ].unique()
                historical_dead[task] = len(previous_dead)

                # Neurons that died in the current task
                current_task_dead = model_data[model_data["death_task"] == task][
                    "neuron_id"
                ].unique()

                # Total dead neurons up to and including this task
                total_dead = np.concatenate([previous_dead, current_task_dead])
                historical_dead[task] = len(np.unique(total_dead))

                # Intersection (persistently dead neurons)
                intersection[task] = len(previous_dead)

        # Calculate newly dead neurons
        new_dead = historical_dead - intersection

        # Plot stacked bars for this model
        plt.bar(
            x_positions + offsets[i],
            new_dead,
            width=width,
            color=new_color,
            alpha=0.7,
            label=f"{model} (New)" if i == 0 else "",
        )

        plt.bar(
            x_positions + offsets[i],
            intersection,
            width=width,
            color=persistent_color,
            bottom=new_dead,
            label=f"{model} (Persistent)" if i == 0 else "",
        )
else:
    print("No timeline data files were found.")
