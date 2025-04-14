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

    def find_persistently_dead_neurons(self, active_task, layer_name, n_epochs=3):
        """
        Identifies neurons that have been consistently dead for the last n_epochs
        across all previous tasks (not just a fixed window).

        Args:
            active_task (int): The current task ID
            layer_name (str): Name of the layer to analyze
            n_epochs (int): Minimum number of consecutive epochs a neuron must be dead in each task

        Returns:
            list: Indices of neurons that have been consistently dead across all previous tasks
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

                # Check if the neuron was dead for at least n_epochs consecutive epochs
                max_consecutive = 0
                current_consecutive = 1

                for i in range(1, len(epochs)):
                    if epochs[i] == epochs[i - 1] + 1:
                        current_consecutive += 1
                    else:
                        max_consecutive = max(max_consecutive, current_consecutive)
                        current_consecutive = 1

                max_consecutive = max(max_consecutive, current_consecutive)

                if max_consecutive < n_epochs:
                    is_persistently_dead = False
                    break

            if is_persistently_dead:
                persistently_dead_neurons.append(neuron_idx)

        print(
            f"\nPersistently Dead Neurons in {layer_name} (across all {len(tasks_to_check)} previous tasks for {n_epochs} epochs each):"
        )
        print(f"Found {len(persistently_dead_neurons)} persistently dead neurons")
        if persistently_dead_neurons:
            print(f"Indices: {persistently_dead_neurons}")

        return persistently_dead_neurons

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
