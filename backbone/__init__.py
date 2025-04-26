from argparse import Namespace
import importlib
import inspect
import os
import math
import csv

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from typing import Callable

from utils import register_dynamic_module_fn

REGISTERED_BACKBONES = (
    dict()
)  # dictionary containing the registered networks. Template: {name: {'class': class, 'parsable_args': parsable_args}}


def xavier(m: nn.Module) -> None:
    """
    Applies Xavier initialization to linear modules.

    Args:
        m: the module to be initialized

    Example::
        >>> net = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        >>> net.apply(xavier)
    """
    if m.__class__.__name__ == "Linear":
        fan_in = m.weight.data.size(1)
        fan_out = m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def num_flat_features(x: torch.Tensor) -> int:
    """
    Computes the total number of items except the first (batch) dimension.

    Args:
        x: input tensor

    Returns:
        number of item from the second dimension onward
    """
    size = x.size()[1:]
    num_features = 1
    for ff in size:
        num_features *= ff
    return num_features


class MammothBackbone(nn.Module):
    """
    A backbone module for the Mammoth model.

    Args:
        **kwargs: additional keyword arguments

    Methods:
        forward: Compute a forward pass.
        features: Get the features of the input tensor (same as forward but with returnt='features').
        get_params: Returns all the parameters concatenated in a single tensor.
        set_params: Sets the parameters to a given value.
        get_grads: Returns all the gradients concatenated in a single tensor.
        get_grads_list: Returns a list containing the gradients (a tensor for each layer).
    """

    def __init__(self, **kwargs) -> None:
        super(MammothBackbone, self).__init__()
        self.device = (
            torch.device("cpu") if "device" not in kwargs else kwargs["device"]
        )
        self.activations = {}
        self.dead_neuron_history = {}
        self.current_task = 0
        self.current_epoch = 0
        self.overlap_ratios = {}

        self.optimizer_name = kwargs.get("optimizer", "adam")
        self.learning_rate = kwargs.get("lr", 0.001)

        self.csv_filename_reviving_neurons = (
            f"results/death_rates_resnet18/death_rate-0.3/dead_neuron_overlap_"
            f"opt_{self.optimizer_name}_"
            f"lr_{self.learning_rate}.csv"
            # f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    def to(self, device, *args, **kwargs):
        super(MammothBackbone, self).to(device, *args, **kwargs)
        self.device = device
        return self

    def forward(self, x: torch.Tensor, returnt="out") -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type ('out' or 'features')
        Returns:
            output tensor or features dictionary
        """
        raise NotImplementedError

    def save_combined_neuron_analysis(
        self, active_task, output_path="results/new_neuron_analysis", create_new=False
    ):
        """
        Save comprehensive neuron analysis data to CSV files, combining all layers.
        This function appends to existing files if they exist, or creates new ones otherwise.

        Args:
            active_task (int): Current task ID
            output_path (str): Directory to save the CSV files
            create_new (bool): Force creation of new files even if they exist

        Returns:
            tuple: Paths to the CSV files (dead_neurons_file, reviving_neurons_file)
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Fixed filenames without timestamp for consistent appending
        dead_neurons_file = os.path.join(output_path, f"all_layers_dead_neurons.csv")
        reviving_neurons_file = os.path.join(
            output_path, f"all_layers_reviving_neurons.csv"
        )

        # Check if files already exist
        dead_neurons_exists = os.path.exists(dead_neurons_file) and not create_new
        reviving_neurons_exists = (
            os.path.exists(reviving_neurons_file) and not create_new
        )

        # 1. Combined dead neurons data across all layers
        all_dead_neurons_data = []

        # 2. Combined reviving neurons data across all layers
        all_reviving_neurons_data = []

        # Analyze each layer with history
        for layer_name in self.dead_neuron_history.keys():
            print(f"Analyzing layer: {layer_name}")

            # Get persistently dead neurons
            consecutive_data = self.find_persistently_dead_neurons_accross_tasks(
                active_task, layer_name, n_tasks=2, n_epochs=5
            )

            if consecutive_data and "consecutively_dead_neurons" in consecutive_data:
                for neuron_idx in consecutive_data["consecutively_dead_neurons"]:
                    # Get neuron history
                    history = self.dead_neuron_history[layer_name][neuron_idx]
                    first_task = min(history.keys())
                    first_epoch = min(history[first_task])
                    last_task = max(history.keys())
                    last_epoch = max(history[last_task])
                    task_count = len(history.keys())
                    epoch_count = sum(len(epochs) for epochs in history.values())

                    # Get consecutive tasks data
                    longest_streak = 0
                    streak_start = None
                    streak_end = None

                    # Check if this neuron has consecutive death info
                    if neuron_idx in consecutive_data.get("death_info", {}):
                        info = consecutive_data["death_info"][neuron_idx]
                        longest_streak = info["consecutive_tasks"]
                        streak_start = info["longest_streak_start"]
                        streak_end = streak_start + longest_streak - 1

                    still_dead = active_task - 1 in history

                    # Add to combined data
                    all_dead_neurons_data.append(
                        {
                            "task": active_task,  # Add current task for tracking when this data was collected
                            "layer_name": layer_name,
                            "neuron_id": neuron_idx,
                            "first_death_task": first_task,
                            "first_death_epoch": first_epoch,
                            "last_death_task": last_task,
                            "last_death_epoch": last_epoch,
                            "task_count": task_count,
                            "epoch_count": epoch_count,
                            "longest_streak": longest_streak if longest_streak else 0,
                            "streak_start": (
                                streak_start if streak_start is not None else -1
                            ),
                            "streak_end": streak_end if streak_end is not None else -1,
                            "still_dead": "Yes" if still_dead else "No",
                        }
                    )

            # Get persistently reviving neurons
            reviving_result = self.find_persistently_reviving_neurons(
                active_task, layer_name, n_epochs=5, min_revive_ratio=0.3
            )

            # Process reviving neurons - no need to call the function twice
            if isinstance(reviving_result, dict) and reviving_result.get(
                "persistently_reviving_neurons"
            ):
                persistently_reviving = reviving_result["persistently_reviving_neurons"]

                for neuron_idx in persistently_reviving:
                    # Get stats from the result dictionary
                    reviving_ratio = reviving_result["reviving_ratios"].get(
                        neuron_idx, 0
                    )
                    revive_count = reviving_result["revive_counts"].get(neuron_idx, 0)
                    death_count = reviving_result["death_counts"].get(neuron_idx, 0)

                    # Get first death
                    history = self.dead_neuron_history[layer_name][neuron_idx]
                    first_task = min(history.keys())
                    first_epoch = min(history[first_task])

                    all_reviving_neurons_data.append(
                        {
                            "task": active_task,  # Add current task for tracking when this data was collected
                            "layer_name": layer_name,
                            "neuron_id": neuron_idx,
                            "first_death_task": first_task,
                            "first_death_epoch": first_epoch,
                            "reviving_ratio": float(reviving_ratio),
                            "revive_count": int(revive_count),
                            "death_count": int(death_count),
                            "optimizer": self.optimizer_name,
                            "learning_rate": self.learning_rate,
                        }
                    )

        # Save or append combined dead neurons data
        if all_dead_neurons_data:
            # Define fieldnames for dead neurons CSV
            dead_fieldnames = [
                "task",
                "layer_name",
                "neuron_id",
                "first_death_task",
                "first_death_epoch",
                "last_death_task",
                "last_death_epoch",
                "task_count",
                "epoch_count",
                "longest_streak",
                "streak_start",
                "streak_end",
                "still_dead",
            ]

            # Append to existing file or create new one
            with open(
                dead_neurons_file, "a" if dead_neurons_exists else "w", newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=dead_fieldnames)
                if not dead_neurons_exists:
                    writer.writeheader()
                writer.writerows(all_dead_neurons_data)

            print(
                f"{'Appended to' if dead_neurons_exists else 'Created'} dead neuron data file: {dead_neurons_file}"
            )

        # Save or append combined reviving neurons data
        if all_reviving_neurons_data:
            # Define fieldnames for reviving neurons CSV
            reviving_fieldnames = [
                "task",
                "layer_name",
                "neuron_id",
                "first_death_task",
                "first_death_epoch",
                "reviving_ratio",
                "revive_count",
                "death_count",
                "optimizer",
                "learning_rate",
            ]

            # Append to existing file or create new one
            with open(
                reviving_neurons_file,
                "a" if reviving_neurons_exists else "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(f, fieldnames=reviving_fieldnames)
                if not reviving_neurons_exists:
                    writer.writeheader()
                writer.writerows(all_reviving_neurons_data)

            print(
                f"{'Appended to' if reviving_neurons_exists else 'Created'} reviving neuron data file: {reviving_neurons_file}"
            )

        return (dead_neurons_file, reviving_neurons_file)

    def track_dead_neurons(self, layer_name, dead_neurons, task_id, epoch):
        """
        Track dead neurons across tasks and epochs.

        Args:
            layer_name (str): Name of the layer
            dead_neurons (np.ndarray): Indices of dead neurons
            task_id (int): Current task ID
            epoch (int): Current epoch
        """
        if layer_name not in self.dead_neuron_history:
            self.dead_neuron_history[layer_name] = {}

        for neuron_idx in dead_neurons:
            if neuron_idx not in self.dead_neuron_history[layer_name]:
                self.dead_neuron_history[layer_name][neuron_idx] = {task_id: [epoch]}
            else:
                if task_id not in self.dead_neuron_history[layer_name][neuron_idx]:
                    self.dead_neuron_history[layer_name][neuron_idx][task_id] = [epoch]
                else:
                    self.dead_neuron_history[layer_name][neuron_idx][task_id].append(
                        epoch
                    )

    def find_persistently_dead_neurons(self, active_task, layer_name, n_epochs=10):
        """
        Identifies neurons that have been consistently dead for the last n_epochs
        consecutive epochs in all previous tasks.

        Args:
            active_task (int): The current task ID
            layer_name (str): Name of the layer to analyze
            n_epochs (int): Number of consecutive epochs at the end of each task to check

        Returns:
            list: Indices of neurons that have been consistently dead in the final epochs across all previous tasks
        """
        if layer_name not in self.dead_neuron_history or active_task <= 0:
            print(f"Current task: {active_task}, need at least one previous task.")
            return []

        # Get all neurons in this layer
        all_neurons = set(self.dead_neuron_history[layer_name].keys())

        # Check all previous tasks
        tasks_to_check = list(range(0, active_task))

        persistently_dead_neurons = []

        for neuron_idx in all_neurons:
            history = self.dead_neuron_history[layer_name].get(neuron_idx, {})
            is_persistently_dead = True

            for task_id in tasks_to_check:
                if task_id not in history:
                    is_persistently_dead = False
                    break

                # Get epochs for this task
                epochs = sorted(history[task_id])
                if not epochs:
                    is_persistently_dead = False
                    break

                # Find the maximum epoch for this task
                max_epoch = epochs[-1]

                # Check if the last n_epochs are present
                required_epochs = list(range(max_epoch - n_epochs + 1, max_epoch + 1))

                # Verify neuron was dead in the last n consecutive epochs
                for req_epoch in required_epochs:
                    if req_epoch not in epochs:
                        is_persistently_dead = False
                        break

                # If this task failed the check, no need to continue
                if not is_persistently_dead:
                    break

            if is_persistently_dead:
                persistently_dead_neurons.append(neuron_idx)

        print(
            f"\nPersistently Dead Neurons in {layer_name} (across all {len(tasks_to_check)} previous tasks for final {n_epochs} epochs each):"
        )
        print(f"Found {len(persistently_dead_neurons)} persistently dead neurons")
        if persistently_dead_neurons:
            print(f"Indices: {persistently_dead_neurons}")

        return persistently_dead_neurons

    def find_persistently_dead_neurons_accross_tasks(
        self, active_task, layer_name, n_tasks=2, n_epochs=5
    ):
        """
        Identifies neurons that have been consistently dead for at least n_tasks consecutive tasks
        starting from when they first died, checking the final n_epochs of each task.

        Args:
            active_task (int): The current task ID
            layer_name (str): Name of the layer to analyze
            n_tasks (int): Minimum number of consecutive tasks required to be dead
            n_epochs (int): Number of consecutive epochs at the end of each task to check
        Returns:
            dict: Dictionary with consecutively dead neurons and their death information
        """
        if layer_name not in self.dead_neuron_history:
            print(f"No dead neuron history for layer {layer_name}")
            return {"consecutively_dead_neurons": [], "death_info": {}}

        # Get all neurons in this layer
        all_neurons = set(self.dead_neuron_history[layer_name].keys())

        consecutively_dead = []
        death_info = {}

        for neuron_idx in all_neurons:
            history = self.dead_neuron_history[layer_name].get(neuron_idx, {})
            if not history:
                continue

            # Find first task where neuron died
            first_task = min(history.keys())

            # Count consecutive tasks where neuron remained dead
            consecutive_tasks = 0
            current_streak = 0
            longest_streak_start = first_task

            for task_id in range(first_task, active_task):
                # Check if neuron was consistently dead in this task
                if task_id in history:
                    # Get epochs for this task
                    epochs = sorted(history[task_id])
                    if not epochs:
                        # Break current streak
                        if current_streak > consecutive_tasks:
                            consecutive_tasks = current_streak
                            longest_streak_start = task_id - current_streak
                        current_streak = 0
                        continue

                    # Find the maximum epoch for this task
                    max_epoch = max(epochs)

                    # Check if the last n_epochs are present
                    required_epochs = list(
                        range(max_epoch - n_epochs + 1, max_epoch + 1)
                    )
                    all_required_epochs_present = True

                    for req_epoch in required_epochs:
                        if req_epoch not in epochs:
                            all_required_epochs_present = False
                            break

                    if all_required_epochs_present:
                        current_streak += 1
                    else:
                        # Break current streak
                        if current_streak > consecutive_tasks:
                            consecutive_tasks = current_streak
                            longest_streak_start = task_id - current_streak
                        current_streak = 0
                else:
                    # Break current streak
                    if current_streak > consecutive_tasks:
                        consecutive_tasks = current_streak
                        longest_streak_start = task_id - current_streak
                    current_streak = 0

            # Check final streak
            if current_streak > consecutive_tasks:
                consecutive_tasks = current_streak
                longest_streak_start = active_task - current_streak

            # Only consider neurons that remained dead for at least n_tasks consecutive tasks
            if consecutive_tasks >= n_tasks:
                consecutively_dead.append(neuron_idx)

                # Record death information
                first_death_task = min(history.keys())
                first_death_epoch = min(history[first_death_task])
                death_info[neuron_idx] = {
                    "first_death": (first_death_task, first_death_epoch),
                    "longest_streak_start": longest_streak_start,
                    "consecutive_tasks": consecutive_tasks,
                    "death_span": list(
                        range(
                            longest_streak_start,
                            longest_streak_start + consecutive_tasks,
                        )
                    ),
                }

        # Print analysis results
        print(f"\nConsecutively Dead Neurons Analysis for {layer_name}:")
        print(f"Total neurons analyzed: {len(all_neurons)}")
        print(
            f"Found {len(consecutively_dead)} neurons that remained dead for at least {n_tasks} consecutive tasks"
        )

        if consecutively_dead:
            print("\nDetails of consecutively dead neurons:")
            for neuron_idx in consecutively_dead:
                info = death_info[neuron_idx]
                first_task, first_epoch = info["first_death"]
                death_span = info["death_span"]

                print(f"  Neuron {neuron_idx}:")
                print(f"    First died at: Task {first_task}, Epoch {first_epoch}")
                print(
                    f"    Longest streak: {info['consecutive_tasks']} consecutive tasks (tasks {death_span[0]}-{death_span[-1]})"
                )

                # Check if still dead in current task
                is_still_dead = active_task - 1 in history
                print(
                    f"    Still dead in most recent task: {'Yes' if is_still_dead else 'No'}"
                )

        return {
            "consecutively_dead_neurons": consecutively_dead,
            "death_info": death_info,
        }

    def find_permanently_dead_neurons(self, layer_name):
        """
        Identifies neurons that die and remain dead throughout the training process.

        Args:
            layer_name (str): Name of the layer to analyze

        Returns:
            dict: A dictionary containing permanently dead neurons and their death statistics
                {
                    'permanent_dead_neurons': list of neuron indices,
                    'death_points': dict mapping neuron index to (task_id, epoch) when it died,
                    'total_neurons_analyzed': total number of neurons tracked
                }
        """
        if layer_name not in self.dead_neuron_history:
            return {
                "permanent_dead_neurons": [],
                "death_points": {},
                "total_neurons_analyzed": 0,
            }

        # Get all tasks in chronological order
        all_tasks = set()
        for neuron_data in self.dead_neuron_history[layer_name].values():
            all_tasks.update(neuron_data.keys())
        all_tasks = sorted(list(all_tasks))

        permanent_dead = []
        death_points = {}

        # Analyze each neuron's death pattern
        for neuron_idx, history in self.dead_neuron_history[layer_name].items():
            if not history:  # Skip if no history
                continue

            # Find the first death point
            first_death_task = min(history.keys())
            first_death_epoch = min(history[first_death_task])

            # Check if the neuron stays dead after its first death
            stays_dead = True
            for task_id in all_tasks:
                if task_id < first_death_task:
                    continue

                if task_id not in history:
                    # If we don't find the neuron in later tasks, it's not permanently dead
                    stays_dead = False
                    break

                # For the first death task, check only epochs after death
                if task_id == first_death_task:
                    all_epochs_after_death = True
                    for epoch in range(
                        first_death_epoch + 1, max(history[task_id]) + 1
                    ):
                        if epoch not in history[task_id]:
                            all_epochs_after_death = False
                            break
                    if not all_epochs_after_death:
                        stays_dead = False
                        break
                else:
                    # For subsequent tasks, the neuron should be dead in all epochs
                    if len(history[task_id]) == 0:
                        stays_dead = False
                        break

            if stays_dead:
                permanent_dead.append(neuron_idx)
                death_points[neuron_idx] = (first_death_task, first_death_epoch)

        result = {
            "permanent_dead_neurons": permanent_dead,
            "death_points": death_points,
            "total_neurons_analyzed": len(self.dead_neuron_history[layer_name]),
        }

        # Print analysis results
        print(f"\nPermanently Dead Neurons Analysis for {layer_name}:")
        print(f"Total neurons tracked: {result['total_neurons_analyzed']}")
        print(f"Number of permanently dead neurons: {len(permanent_dead)}")
        if permanent_dead:
            print("\nPermanently dead neurons and their death points:")
            for neuron_idx in permanent_dead:
                task, epoch = death_points[neuron_idx]
                print(f"Neuron {neuron_idx}: Died at Task {task}, Epoch {epoch}")

        return result

    def save_dead_neuron_data(
        self, layer_name, output_path="results/death_rates_resnet18/death_rate-0.3"
    ):
        """
        Save dead neuron analysis results to CSV files for plotting.

        Args:
            layer_name (str): Name of the layer to analyze
            output_path (str): Directory to save the CSV files
        """
        result = self.find_permanently_dead_neurons(layer_name)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Prepare data for death timeline
        timeline_data = []
        for neuron_idx in result["permanent_dead_neurons"]:
            task, epoch = result["death_points"][neuron_idx]
            timeline_data.append(
                {"neuron_id": neuron_idx, "death_task": task, "death_epoch": epoch}
            )

        # Save death timeline data
        timeline_file = os.path.join(
            output_path,
            f"{layer_name}_death_timeline_"
            f"opt_{self.optimizer_name}_"
            f"lr_{self.learning_rate}.csv",
            # f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        with open(timeline_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["neuron_id", "death_task", "death_epoch"]
            )
            writer.writeheader()
            writer.writerows(timeline_data)

        # Prepare and save death distribution data
        distribution_data = {}
        for neuron_idx in result["permanent_dead_neurons"]:
            task, epoch = result["death_points"][neuron_idx]
            key = (task, epoch)
            distribution_data[key] = distribution_data.get(key, 0) + 1

        distribution_file = os.path.join(
            output_path,
            f"{layer_name}_death_distribution_"
            f"opt_{self.optimizer_name}_"
            f"lr_{self.learning_rate}.csv",
        )
        with open(distribution_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task", "epoch", "death_count"])
            writer.writeheader()
            for (task, epoch), count in distribution_data.items():
                writer.writerow({"task": task, "epoch": epoch, "death_count": count})

    def calculate_dead_neuron_overlap(self, layer_name, task_id, epoch):
        """
        Calculate the overlap ratio between current dead neurons and historical dead neurons.

        Args:
            layer_name (str): Name of the layer to analyze
            task_id (int): Current task ID
            epoch (int): Current epoch

        Returns:
            float: Overlap ratio (|X ∩ Y| / min(|X|, |Y|)) between historical and current dead neurons
        """
        if layer_name not in self.dead_neuron_history:
            return 0.0

        # Get current dead neurons for this task and epoch
        current_dead = set()
        for neuron_idx, history in self.dead_neuron_history[layer_name].items():
            if task_id in history:
                # Make sure history[task_id] is a list before checking if epoch is in it
                epochs = history[task_id]
                if isinstance(epochs, list) and epoch in epochs:
                    current_dead.add(neuron_idx)

        # Get historical dead neurons (from all previous epochs and tasks)
        historical_dead = set()
        for neuron_idx, history in self.dead_neuron_history[layer_name].items():
            for prev_task in history.keys():
                if prev_task < task_id:
                    historical_dead.add(neuron_idx)
                elif prev_task == task_id:
                    # Make sure history[prev_task] is a list before checking epochs
                    epochs = history[prev_task]
                    if isinstance(epochs, list) and any(e < epoch for e in epochs):
                        historical_dead.add(neuron_idx)

        # Calculate overlap ratio
        if not current_dead or not historical_dead:
            return 0.0

        intersection = len(current_dead.intersection(historical_dead))
        min_size = min(len(current_dead), len(historical_dead))
        overlap_ratio = intersection / min_size

        # Store the results
        if layer_name not in self.overlap_ratios:
            self.overlap_ratios[layer_name] = []

        result = {
            "layer": layer_name,
            "task": task_id,
            "epoch": epoch,
            "overlap_ratio": overlap_ratio,
            "current_dead": len(current_dead),
            "historical_dead": len(historical_dead),
            "intersection": intersection,
        }
        self.overlap_ratios[layer_name].append(result)

        # Save to CSV
        os.makedirs(os.path.dirname(self.csv_filename_reviving_neurons), exist_ok=True)
        csv_exists = os.path.exists(self.csv_filename_reviving_neurons)
        with open(self.csv_filename_reviving_neurons, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not csv_exists:
                writer.writeheader()
            writer.writerow(result)

        return overlap_ratio

    def find_persistently_reviving_neurons(
        self, active_task, layer_name, n_epochs=5, min_revive_ratio=0.3
    ):
        """
        Identifies neurons that consistently revive within the last n_epochs of each task.

        A neuron is considered "persistently reviving" if it shows revival patterns in the
        final n_epochs of each previous task with a revival ratio exceeding the threshold.

        Args:
            active_task (int): The current task ID
            layer_name (str): Name of the layer to analyze
            n_epochs (int): Number of consecutive epochs at the end of each task to check
            min_revive_ratio (float): Minimum ratio of revivals to deaths required
                                to consider a neuron as persistently reviving

        Returns:
            dict: Dictionary containing reviving statistics for each neuron within the window:
                {
                    'persistently_reviving_neurons': list of neuron indices,
                    'reviving_ratios': dict mapping neuron index to its reviving ratio,
                    'revive_counts': dict mapping neuron index to revival count,
                    'death_counts': dict mapping neuron index to death count
                }
        """
        if layer_name not in self.dead_neuron_history or active_task <= 0:
            print(f"Current task: {active_task}, need at least one previous task.")
            return {
                "persistently_reviving_neurons": [],
                "reviving_ratios": {},
                "revive_counts": {},
                "death_counts": {},
            }

        # Get all neurons in this layer
        all_neurons = set(self.dead_neuron_history[layer_name].keys())

        # Check all previous tasks
        tasks_to_check = list(range(0, active_task))

        # Initialize statistics dictionaries
        reviving_stats = {}

        for neuron_idx in all_neurons:
            history = self.dead_neuron_history[layer_name].get(neuron_idx, {})
            if not history:
                continue

            # Track across all previous tasks
            total_death_count = 0
            total_revive_count = 0
            is_consistently_reviving = True

            for task_id in tasks_to_check:
                if task_id not in history:
                    continue

                # Get epochs for this task
                all_epochs = sorted(history[task_id])
                if not all_epochs:
                    continue

                # Find the last n_epochs for this task
                max_epoch = max(all_epochs)
                window_start = max_epoch - n_epochs + 1
                window_end = max_epoch
                window_epochs = [
                    e for e in all_epochs if window_start <= e <= window_end
                ]

                if (
                    not window_epochs or len(window_epochs) < 2
                ):  # Need at least 2 epochs to detect revival
                    continue

                # Count deaths and revivals within this window
                death_count = 0
                revive_count = 0

                # First epoch in window may be a death if not already dead before
                was_dead = False
                for i, epoch in enumerate(window_epochs):
                    if i == 0:
                        # First epoch in window is a death
                        was_dead = True
                        death_count += 1
                        continue

                    # Check for gaps (potentially revived and died again)
                    prev_epoch = window_epochs[i - 1]
                    if epoch > prev_epoch + 1:
                        # Gap indicates revival followed by death
                        revive_count += 1
                        death_count += 1

                # Calculate revival ratio for this task
                task_revive_ratio = revive_count / max(death_count, 1)

                # Track totals
                total_death_count += death_count
                total_revive_count += revive_count

                # A task needs to have some revival to count as consistently reviving
                if task_revive_ratio < min_revive_ratio:
                    is_consistently_reviving = False

            # Calculate overall revival ratio
            overall_revive_ratio = total_revive_count / max(total_death_count, 1)

            # Store statistics for this neuron
            reviving_stats[neuron_idx] = {
                "reviving_ratio": overall_revive_ratio,
                "revive_count": total_revive_count,
                "death_count": total_death_count,
                "consistently_reviving": is_consistently_reviving
                and total_death_count > 0,
            }

        # Identify persistently reviving neurons
        persistently_reviving = []
        reviving_ratios = {}
        revive_counts = {}
        death_counts = {}

        for neuron_idx, stats in reviving_stats.items():
            if (
                stats["consistently_reviving"]
                and stats["reviving_ratio"] >= min_revive_ratio
            ):
                persistently_reviving.append(neuron_idx)
                reviving_ratios[neuron_idx] = stats["reviving_ratio"]
                revive_counts[neuron_idx] = stats["revive_count"]
                death_counts[neuron_idx] = stats["death_count"]

        # Sort by reviving ratio (highest first)
        persistently_reviving = sorted(
            persistently_reviving, key=lambda n: reviving_ratios[n], reverse=True
        )

        # Print analysis results
        print(
            f"\nPersistently Reviving Neurons Analysis for {layer_name} (in final {n_epochs} epochs of each task):"
        )
        print(f"Total neurons analyzed: {len(all_neurons)}")
        print(
            f"Found {len(persistently_reviving)} neurons that persistently revive (ratio ≥ {min_revive_ratio})"
        )

        if persistently_reviving:
            print("\nTop 10 most frequently reviving neurons:")
            for idx, neuron in enumerate(persistently_reviving[:10]):
                print(
                    f"  {idx+1}. Neuron {neuron}: "
                    f"Reviving ratio {reviving_ratios[neuron]:.2f} "
                    f"({revive_counts[neuron]} revivals / {death_counts[neuron]} deaths)"
                )

        result = {
            "persistently_reviving_neurons": persistently_reviving,
            "reviving_ratios": reviving_ratios,
            "revive_counts": revive_counts,
            "death_counts": death_counts,
        }

        return result

    def get_reviving_neurons(
        self, layer_name, task_id=None, current_epoch=None, prev_epoch=None
    ):
        """
        Returns the indices of neurons that have revived (were dead previously but are now active).
        Can compare between tasks or between specific epochs within the same task.

        Args:
            layer_name (str): Name of the layer to analyze
            task_id (int, optional): Current task ID. If None, uses the current task.
            current_epoch (int, optional): Current epoch for comparison. If None, considers all epochs in current task.
            prev_epoch (int, optional): Previous epoch for comparison. If None, uses previous task's epochs.

        Returns:
            list: Indices of reviving neurons
        """
        if layer_name not in self.dead_neuron_history:
            return []

        if task_id is None:
            task_id = self.current_task

        # Case 1: Comparing epochs within the same task
        if current_epoch is not None and prev_epoch is not None and task_id > 0:
            # Get currently dead neurons for this specific epoch
            current_dead = set()
            for neuron_idx, history in self.dead_neuron_history[layer_name].items():
                if task_id in history and current_epoch in history[task_id]:
                    current_dead.add(neuron_idx)

            # Get previously dead neurons for the specific previous epoch
            prev_dead = set()
            for neuron_idx, history in self.dead_neuron_history[layer_name].items():
                if task_id in history and prev_epoch in history[task_id]:
                    prev_dead.add(neuron_idx)

            # Reviving neurons are those that were dead previously but now active
            reviving_neurons = sorted([n for n in prev_dead if n not in current_dead])

            print(
                f"\nReviving Neurons Analysis for {layer_name} at Task {task_id} between epochs {prev_epoch}->{current_epoch}:"
            )
            print(f"Previously dead neurons (epoch {prev_epoch}): {len(prev_dead)}")
            print(
                f"Currently dead neurons (epoch {current_epoch}): {len(current_dead)}"
            )
            print(f"Number of reviving neurons: {len(reviving_neurons)}")

            if reviving_neurons:
                print(f"Reviving neurons: {reviving_neurons}")

            return reviving_neurons

        # Case 2: Comparing between tasks (original functionality)
        elif task_id > 0:
            # Get current dead neurons
            current_dead = set()
            for neuron_idx, history in self.dead_neuron_history[layer_name].items():
                if task_id in history:
                    current_dead.add(neuron_idx)

            # Get previously dead neurons
            prev_dead = set()
            for neuron_idx, history in self.dead_neuron_history[layer_name].items():
                if task_id - 1 in history:
                    prev_dead.add(neuron_idx)

            # Initialize the revived neuron history for this layer if it doesn't exist
            if not hasattr(self, "revived_neuron_history"):
                self.revived_neuron_history = {}

            if layer_name not in self.revived_neuron_history:
                self.revived_neuron_history[layer_name] = {}

            # Reviving neurons are those that were dead in the previous task,
            # are now active (not in current_dead), and haven't been revived before
            reviving_neurons = []
            for n in prev_dead:
                if n not in current_dead:
                    # Check if this neuron has already been revived in the current task
                    if n not in self.revived_neuron_history[layer_name].get(
                        task_id, {}
                    ):
                        reviving_neurons.append(n)

            reviving_neurons = sorted(reviving_neurons)

            # Print analysis results
            print(
                f"\nReviving Neurons Analysis for {layer_name} between Task {task_id-1} and Task {task_id}:"
            )
            print(f"Previously dead neurons: {len(prev_dead)}")
            print(f"Currently dead neurons: {len(current_dead)}")
            print(f"Number of reviving neurons: {len(reviving_neurons)}")

            # Print the list of neurons if there are any
            if reviving_neurons:
                print(f"Reviving neurons in {layer_name}: {reviving_neurons}")

            return reviving_neurons

        return []

    def analyze_dead_neuron_consistency(self, layer_name):
        """
        Analyze consistency of dead neurons across tasks and epochs.
        Shows only when neurons were dead and for how long.
        """
        if layer_name not in self.dead_neuron_history:
            print(f"No dead neuron history for layer {layer_name}")
            return

        print(f"\nDead Neuron Consistency Analysis for {layer_name}:")

        for neuron_idx, history in self.dead_neuron_history[layer_name].items():
            tasks = sorted(history.keys())

            # Analyze consecutive dead periods
            consecutive_periods = []
            current_period = None

            for task_id in tasks:
                epochs = sorted(history[task_id])

                # Check for continuous epochs
                for epoch in epochs:
                    if current_period is None:
                        current_period = {
                            "start_task": task_id,
                            "start_epoch": epoch,
                            "end_task": task_id,
                            "end_epoch": epoch,
                        }
                    elif (
                        task_id == current_period["end_task"]
                        and epoch == current_period["end_epoch"] + 1
                    ):
                        # Continuous in same task
                        current_period["end_epoch"] = epoch
                    # elif task_id == current_period["end_task"] + 1:
                    #     # Continuous across tasks
                    #     current_period["end_task"] = task_id
                    #     current_period["end_epoch"] = epoch
                    else:
                        # Gap found, store current period and start new one
                        consecutive_periods.append(current_period)
                        current_period = {
                            "start_task": task_id,
                            "start_epoch": epoch,
                            "end_task": task_id,
                            "end_epoch": epoch,
                        }

            if current_period is not None:
                consecutive_periods.append(current_period)

            # Print analysis for this neuron
            if consecutive_periods:
                # print(f"\nNeuron {neuron_idx}:")
                for period in consecutive_periods:
                    if period["start_task"] == period["end_task"]:
                        duration = period["end_epoch"] - period["start_epoch"] + 1
                        # print(
                        #     f"  Dead in Task {period['start_task']}: "
                        #     f"Epochs {period['start_epoch']}-{period['end_epoch']} "
                        #     f"(Duration: {duration} epochs)"
                        # )
                    else:
                        total_epochs = sum(
                            len(history[t])
                            for t in range(period["start_task"], period["end_task"] + 1)
                        )
                        # print(
                        #     f"  Dead from Task {period['start_task']} "
                        #     f"Epoch {period['start_epoch']} to Task "
                        #     f"{period['end_task']} Epoch {period['end_epoch']} "
                        #     f"(Total epochs dead: {total_epochs})"
                        # )

    def check_inactive_neurons(self, threshold=0.1, show_histogram=False):
        """
        Analyze inactive neurons in each layer based on the stored activations.
        A neuron is considered inactive if its mean activation value is below the threshold.

        Args:
            threshold (float): threshold value to consider a neuron inactive (default: 1e-3)

        Returns:
            dict: Dictionary containing inactive neuron statistics for each layer
        """
        inactive_stats = {}

        if show_histogram:
            import matplotlib.pyplot as plt

            fig_size = (15, 10)
            num_layers = len(self.activations)
            fig, axs = plt.subplots(num_layers, 1, figsize=fig_size, squeeze=False)
            layer_index = 0

        for layer_name, activation in self.activations.items():
            # Move activation to CPU and convert to numpy for analysis
            act_np = activation.detach().cpu().numpy()

            # Reshape to (batch_size, channels, -1) to consider spatial dimensions together
            batch_size, channels = act_np.shape[0], act_np.shape[1]
            reshaped_act = act_np.reshape(batch_size, channels, -1)

            # Calculate statistics per channel
            mean_activations = np.mean(
                np.abs(reshaped_act), axis=(0, 2)
            )  # Average across batch and spatial dims
            max_activations = np.max(
                np.abs(reshaped_act), axis=(0, 2)
            )  # Max across batch and spatial dims

            # Consider a neuron inactive if its mean activation is below threshold
            inactive_mask = mean_activations <= threshold
            inactive_count = inactive_mask.sum()
            total_channels = channels

            dead_neurons = (reshaped_act == 0).all(axis=(0, 2))
            dead_count = dead_neurons.sum()

            # Get activation statistics
            stats = {
                "total_channels": total_channels,
                "dead_count": int(dead_count),
                "dead_percentage": (dead_count / channels) * 100,
                "inactive_count": int(inactive_count),
                "inactive_percentage": (inactive_count / total_channels) * 100,
                "mean_activation": float(np.mean(mean_activations)),
                "std_activation": float(np.std(mean_activations)),
                "max_activation": float(np.max(max_activations)),
                "min_activation": float(np.min(mean_activations)),
            }

            inactive_stats[layer_name] = stats

            # Print detailed statistics
            print(f"\nLayer: {layer_name}")
            print(f"Shape: {activation.shape}")
            print(
                f"Dead neurons: {stats['dead_count']}/{stats['total_channels']} ({stats['dead_percentage']:.2f}%)"
            )
            print(
                f"Inactive neurons: {stats['inactive_count']}/{stats['total_channels']} ({stats['inactive_percentage']:.2f}%)"
            )
            print(f"Mean activation: {stats['mean_activation']:.6f}")
            print(f"Std activation: {stats['std_activation']:.6f}")
            print(f"Min activation: {stats['min_activation']:.6f}")
            print(f"Max activation: {stats['max_activation']:.6f}")

            if show_histogram:
                ax = axs[layer_index, 0]

                all_activations = reshaped_act.flatten()

                max_samples = 100000
                if len(all_activations) > max_samples:
                    indices = np.random.choice(
                        len(all_activations), max_samples, replace=False
                    )
                    all_activations = all_activations[indices]

                # Plot histogram of mean activations
                counts, edges, _ = ax.hist(
                    (mean_activations - np.min(mean_activations))
                    / (np.max(mean_activations) - np.min(mean_activations) + 1e-10),
                    bins=50,
                    alpha=0.7,
                    density=False,
                    color="steelblue",
                )

                # Draw vertical line at threshold
                ax.axvline(
                    x=threshold,
                    color="red",
                    linestyle="--",
                    label=f"Threshold ({threshold})",
                )

                # Highlight inactive neuron region
                ax.axvspan(
                    0,
                    threshold,
                    alpha=0.2,
                    color="red",
                    label=f'Inactive: {inactive_count}/{channels} ({stats["inactive_percentage"]:.1f}%)',
                )

                # Add title and labels
                ax.set_title(f"Layer: {layer_name} - Activation Distribution")
                ax.set_xlabel("Mean Absolute Activation")
                ax.set_ylabel("Number of Neurons")
                ax.legend()

                # Add text with statistics
                # stats_text = (
                #     f"Total neurons: {channels}\n"
                #     f"Dead neurons: {stats['dead_count']} ({stats['dead_percentage']:.1f}%)\n"
                #     f"Inactive neurons: {stats['inactive_count']} ({stats['inactive_percentage']:.1f}%)\n"
                #     f"Mean: {stats['mean_activation']:.4f}\n"
                #     f"Median: {float(np.median(mean_activations)):.4f}\n"
                #     f"Min: {stats['min_activation']:.4f}\n"
                #     f"Max: {stats['max_activation']:.4f}"
                # )
                # ax.text(
                #     0.98,
                #     0.95,
                #     stats_text,
                #     transform=ax.transAxes,
                #     verticalalignment="top",
                #     horizontalalignment="right",
                #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                # )

                layer_index += 1

        # Finalize and show/save the histogram plot if requested
        if show_histogram:
            plt.tight_layout()
            plt.show()
            # plt.close(fig)

        return inactive_stats

    def analyze_layer_activations(
        self, layer_name, current_task, current_epoch, dead_threshold=0, top_k=10
    ):
        """
        Analyze activations of individual neurons in a specific layer.
        """
        if layer_name not in self.activations:
            print(f"Layer {layer_name} not found in activations")
            return

        # Get activations for the specified layer
        activation = self.activations[layer_name]
        act_np = activation.detach().cpu().numpy()

        # Reshape to (batch_size, channels, -1)
        batch_size, channels = act_np.shape[0], act_np.shape[1]
        reshaped_act = act_np.reshape(batch_size, channels, -1)

        # Calculate different statistics per neuron
        raw_mean = np.mean(reshaped_act, axis=(0, 2))  # Without abs()
        raw_std = np.std(reshaped_act, axis=(0, 2))
        zero_percentage = np.mean(reshaped_act <= dead_threshold, axis=(0, 2)) * 100

        mean_activations = np.mean(np.abs(reshaped_act), axis=(0, 2))
        # Normalize mean activations to [0, 1] range
        normalized_mean_activations = (mean_activations - np.min(mean_activations)) / (
            np.max(mean_activations) - np.min(mean_activations) + 1e-10
        )
        inactive_mask = normalized_mean_activations <= dead_threshold
        inactive_neurons = np.where(inactive_mask)[0]

        print(f"\nLayer: {layer_name}")
        # print("inactive_mask: ", inactive_mask)
        # print("Inactive Neurons: ", np.where(inactive_mask)[0])
        print("Inactive Neurons Count: ", len(inactive_neurons))

        # Sort neurons by raw mean activation
        neuron_indices = np.argsort(np.abs(raw_mean))

        dead_neurons = np.where(zero_percentage >= 99.99)[
            0
        ]  # Using 99.99% threshold to account for numerical precision
        if len(inactive_neurons) > 0:
            # print(f"\nDead neurons found ({len(dead_neurons)}):")
            # print(f"Indices: {dead_neurons.tolist()}")
            # print(f"Percentage: {(len(dead_neurons)/channels)*100:.2f}% of layer")

            self.track_dead_neurons(
                layer_name, inactive_neurons, current_task, current_epoch
            )
            overlap_ratio = self.calculate_dead_neuron_overlap(
                layer_name, current_task, current_epoch
            )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the features of the input tensor.

        Args:
            x: input tensor

        Returns:
            features tensor
        """
        return self.forward(x, returnt="features")

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.

        Returns:
            parameters tensor
        """
        return torch.nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.

        Args:
            new_params: concatenated values to be set
        """
        torch.nn.utils.vector_to_parameters(new_params, self.parameters())

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    def set_grads(self, new_grads: torch.Tensor) -> None:
        """
        Sets the gradients of all parameters.

        Args:
            new_params: concatenated values to be set
        """
        progress = 0
        for pp in list(self.parameters()):
            cand_grads = new_grads[
                progress : progress + torch.tensor(pp.size()).prod()
            ].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.grad = cand_grads


def register_backbone(name: str) -> Callable:
    """
    Decorator to register a backbone network for use in a Dataset. The decorator may be used on a class that inherits from `MammothBackbone` or on a function that returns a `MammothBackbone` instance.
    The registered model can be accessed using the `get_backbone` function and can include additional keyword arguments to be set during parsing.

    The arguments can be inferred by the *signature* of the backbone network's class. The value of the argument is the default value. If the default is set to `Parameter.empty`, the argument is required. If the default is set to `None`, the argument is optional. The type of the argument is inferred from the default value (default is `str`).

    Args:
        name: the name of the backbone network
    """

    return register_dynamic_module_fn(name, REGISTERED_BACKBONES, MammothBackbone)


def get_backbone_class(name: str, return_args=False) -> MammothBackbone:
    """
    Get the backbone network class from the registered networks.

    Args:
        name: the name of the backbone network
        return_args: whether to return the parsable arguments of the backbone network

    Returns:
        the backbone class
    """
    name = name.replace("_", "-").lower()
    assert (
        name in REGISTERED_BACKBONES
    ), f"Attempted to access non-registered network: {name}"
    cl = REGISTERED_BACKBONES[name]["class"]
    if return_args:
        return cl, REGISTERED_BACKBONES[name]["parsable_args"]


def get_backbone(args: Namespace) -> MammothBackbone:
    """
    Build the backbone network from the registered networks.

    Args:
        args: the arguments which contains the `--backbone` attribute and the additional arguments required by the backbone network

    Returns:
        the backbone model
    """
    backbone_class, backbone_args = get_backbone_class(args.backbone, return_args=True)
    missing_args = [arg for arg in backbone_args.keys() if arg not in vars(args)]
    assert (
        len(missing_args) == 0
    ), "Missing arguments for the backbone network: " + ", ".join(missing_args)

    parsed_args = {arg: getattr(args, arg) for arg in backbone_args.keys()}

    return backbone_class(**parsed_args)


# import all files in the backbone folder to register the networks
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py":
        importlib.import_module(f"backbone.{file[:-3]}")
