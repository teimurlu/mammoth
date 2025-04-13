#!/usr/bin/env python3
# filepath: run_experiments.py
import os
import subprocess
import time
from datetime import datetime


def run_experiment(cmd, output_dir, name, seed=0):
    """Run an experiment with the given command and save output to a file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{name}_seed{seed}_{timestamp}.log")

    print(f"Running: {cmd}")
    print(f"Saving output to: {output_file}")

    with open(output_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output to both console and file
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        process.wait()

    return process.returncode


def modify_activation(activation_type):
    """Modify ResNetBlock.py to use the specified activation function"""
    filepath = "backbone/ResNetBlock.py"
    with open(filepath, "r") as f:
        content = f.read()

    # Create backup if it doesn't exist
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)

    # First update the imports
    if activation_type == "relu":
        new_content = content.replace(
            "from torch.nn.functional import avg_pool2d, relu, silu",
            "from torch.nn.functional import avg_pool2d, relu",
        )
        new_content = new_content.replace(
            "from torch.nn.functional import avg_pool2d, relu, leaky_relu",
            "from torch.nn.functional import avg_pool2d, relu",
        )

        # Then update the function calls - specifically looking for "" pattern
        new_content = new_content.replace("silu(", "relu(")
        new_content = new_content.replace("leaky_relu(", "relu(")

    elif activation_type == "silu":
        # Add silu import if needed
        if "silu" not in content:
            new_content = content.replace(
                "from torch.nn.functional import avg_pool2d, relu",
                "from torch.nn.functional import avg_pool2d, relu, silu",
            )
        else:
            new_content = content

        # Replace relu calls with silu
        new_content = new_content.replace("relu(", "silu(")
        new_content = new_content.replace("leaky_relu(", "silu(")

    elif activation_type == "leaky_relu":
        # Add leaky_relu import if needed
        if "leaky_relu" not in content:
            new_content = content.replace(
                "from torch.nn.functional import avg_pool2d, relu",
                "from torch.nn.functional import avg_pool2d, relu, leaky_relu",
            )
        else:
            new_content = content

        # Replace relu calls with leaky_relu
        new_content = new_content.replace("relu(", "leaky_relu(")
        new_content = new_content.replace("silu(", "leaky_relu(")
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

    # Write the modified file
    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Modified {filepath} to use {activation_type}")


def modify_dead_threshold(threshold):
    """Modify sgd_thesis.py to use the specified dead_threshold"""
    filepath = "models/sgd_thesis.py"
    with open(filepath, "r") as f:
        content = f.read()

    # Create backup if it doesn't exist
    backup_path = f"{filepath}.backup"
    with open(backup_path, "w") as f:
        f.write(content)

    # Search for all occurrences of dead_threshold=X.Y
    import re

    pattern = r"dead_threshold=([0-9.]+)"
    matches = re.findall(pattern, content)

    if matches:
        # Replace all occurrences of dead_threshold=X.Y with the new value
        new_content = re.sub(pattern, f"dead_threshold={threshold}", content)

        # Write the modified file
        with open(filepath, "w") as f:
            f.write(new_content)
        print(
            f"Modified {filepath} to use dead_threshold={threshold} ({len(matches)} occurrences)"
        )
    else:
        print(f"Warning: Couldn't find dead_threshold in {filepath}")


def modify_reinitialization_method(use_ghada=False):
    """Modify sgd_thesis.py to use either ours or ghada reinitialization method"""
    filepath = "models/sgd_thesis.py"
    with open(filepath, "r") as f:
        content = f.read()

    # Create backup if it doesn't exist
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)

    if use_ghada:
        method = "method1"
    else:
        method = "method2"

    # Find the line in end_epoch method that calls reinitialize_reviving_neurons_*
    if "self.reinitialize_reviving_neurons_" in content:
        new_content = content.replace(
            "self.reinitialize_reviving_neurons_method2",
            f"self.reinitialize_reviving_neurons_{method}",
        )
        new_content = new_content.replace(
            "self.reinitialize_reviving_neurons_method1",
            f"self.reinitialize_reviving_neurons_{method}",
        )
    else:
        print(f"Warning: Couldn't find reinitialization method in {filepath}")
        new_content = content

    # Write the modified file
    with open(filepath, "w") as f:
        f.write(new_content)

    print(f"Modified {filepath} to use reinitialization method: {method}")


def restore_files():
    """Restore all modified files from backups"""
    for filepath in ["backbone/ResNetBlock.py", "models/sgd_thesis.py"]:
        backup_path = f"{filepath}.backup"
        if os.path.exists(backup_path):
            with open(backup_path, "r") as f:
                original = f.read()
            with open(filepath, "w") as f:
                f.write(original)
            print(f"Restored {filepath} from backup")


def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Seeds to use for all experiments
    seeds = [42]

    # Basic experiment configurations
    for seed in seeds:
        output_dir = f"task_accuracies/seeds/{seed}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Add baseline experiment (default ReLU)
        baseline_cmd = f"python3 main.py --model sgd --dataset seq-cifar10 --optimizer sgd --lr 0.1 --n_epochs 10 --backbone resnet18 --seed {seed}"
        run_experiment(baseline_cmd, output_dir, "01_baseline_sgd_relu", seed=seed)

        # Run all combinations
        for activation in ["relu"]:
            # Modify activation function
            # modify_activation(activation)

            for dead_threshold in [0.1]:
                # Modify dead neuron threshold
                modify_dead_threshold(dead_threshold)

                for use_ghada in [False, True]:  # False = ours, True = ghada
                    # Modify reinitialization method
                    modify_reinitialization_method(use_ghada)

                    reinit_method = "method1" if use_ghada else "method2"

                    # SGD optimizer
                    sgd_cmd = f"python3 main.py --model sgd-thesis --dataset seq-cifar10 --optimizer sgd --lr 0.1 --n_epochs 10 --backbone resnet18 --seed {seed}"
                    exp_name = f"{activation}_{dead_threshold}_{reinit_method}_sgd"
                    run_experiment(sgd_cmd, output_dir, exp_name, seed=seed)

                    # Adam optimizer
                    # adam_cmd = f"python3 main.py --model sgd-thesis --dataset seq-cifar10 --optimizer adam --lr 0.001 --n_epochs 10 --backbone resnet18 --seed {seed}"
                    # exp_name = f"{activation}_{dead_threshold}_{reinit_method}_adam"
                    # run_experiment(adam_cmd, output_dir, exp_name, seed=seed)

    # Restore files to their original state
    restore_files()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted. Restoring files...")
        restore_files()
    except Exception as e:
        print(f"Error: {e}")
        restore_files()
