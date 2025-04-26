"""
This module implements the simplest form of incremental training, i.e., finetuning.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from models.utils.continual_model import ContinualModel
import torch.nn.functional as F
from sam.sam import SAM
from sam.example.utility.bypass_bn import enable_running_stats, disable_running_stats
from sam.example.model.smooth_cross_entropy import smooth_crossentropy


class Sgd_Thesis(ContinualModel):
    """
    Finetuning baseline - simple incremental training.
    """

    NAME = "sgd-thesis"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Sgd_Thesis, self).__init__(
            backbone, loss, args, transform, dataset=dataset
        )
        self.active_task = 0
        self.clip_value = 1.0
        self.base_pruning_rate = 0

        # base_optimizer = torch.optim.SGD
        # self.optimizer = SAM(
        #     self.get_parameters(),
        #     base_optimizer,
        #     adaptive=True,
        # )

    def _get_next_layer_name(self, current_layer_name):
        """
        Helper method to get the next layer given the current layer name.
        """
        layers = ["layer1", "layer2", "layer3", "layer4"]
        try:
            current_idx = layers.index(current_layer_name)
            if current_idx < len(layers) - 1:
                return layers[current_idx + 1]
        except ValueError:
            pass
        return None

    def reinitialize_reviving_neurons_method1(
        self, layer_name, current_epoch=None, prev_epoch=None
    ):
        """
        Re-initializes neurons that were dead but have become active again.
        - Incoming weights: Kaiming normal initialization
        - Outgoing weights: Zeroed out
        - BatchNorm parameters: Reset for reviving neurons

        Args:
            layer_name (str): Name of the layer to reinitialize
            current_epoch (int, optional): Current epoch for comparison
            prev_epoch (int, optional): Previous epoch for comparison
        """
        if self.active_task == 0:
            return

        layer = getattr(self.net, layer_name)

        # Get reviving neurons between tasks or epochs
        # if current_epoch is not None and prev_epoch is not None:
        #     reviving_neurons = self.net.get_reviving_neurons(
        #         layer_name, self.active_task, current_epoch, prev_epoch
        #     )
        # else:
        #     reviving_neurons = self.net.get_reviving_neurons(
        #         layer_name, self.active_task
        #     )

        # if current_epoch is not None and prev_epoch is not None:
        reviving_neurons = self.net.find_persistently_reviving_neurons(
            active_task=self.active_task, layer_name=layer_name, n_epochs=50
        )["persistently_reviving_neurons"]

        if not reviving_neurons:
            return

        print(f"Reviving neurons in {layer_name}: {reviving_neurons}")
        print(
            f"Re-initializing {len(reviving_neurons)} reviving neurons in {layer_name}"
        )

        # Handle convolutional layers and their weights
        if hasattr(layer, "weight"):
            # For convolutional layers
            if len(layer.weight.shape) == 4:
                for neuron_idx in reviving_neurons:
                    layer.weight.data[neuron_idx] = 0.0
                    # Custom handling for incoming weights can be added here.
            # For fully connected layers
            elif len(layer.weight.shape) == 2:
                for neuron_idx in reviving_neurons:
                    layer.weight.data[neuron_idx] = 0.0
                    next_layer_name = self._get_next_layer_name(layer_name)
                    if next_layer_name:
                        next_layer = getattr(self.net, next_layer_name)
                        if (
                            hasattr(next_layer, "weight")
                            and len(next_layer.weight.shape) == 2
                        ):
                            torch.nn.init.kaiming_normal_(
                                next_layer.weight[:, neuron_idx : neuron_idx + 1],
                                mode="fan_in",
                                nonlinearity="relu",
                            )

        # Handle bias parameters if they exist
        if hasattr(layer, "bias") and layer.bias is not None:
            for neuron_idx in reviving_neurons:
                layer.bias.data[neuron_idx] = 0

        # Handle BatchNorm parameters for the corresponding layer
        self._reinitialize_batchnorm_for_neurons(layer_name, reviving_neurons)

    def reinitialize_reviving_neurons_method2(
        self, layer_name, current_epoch=None, prev_epoch=None
    ):
        """
        Re-initializes neurons that were dead but have become active again.
        - Both incoming and outgoing weights: Kaiming normal initialization
        - BatchNorm parameters: Reset for reviving neurons

        Args:
            layer_name (str): Name of the layer to reinitialize
            current_epoch (int, optional): Current epoch for comparison
            prev_epoch (int, optional): Previous epoch for comparison
        """
        if self.active_task == 0:
            return

        layer = getattr(self.net, layer_name)

        # Get reviving neurons between tasks or epochs
        # if current_epoch is not None and prev_epoch is not None:
        #     reviving_neurons = self.net.get_reviving_neurons(
        #         layer_name, self.active_task, current_epoch, prev_epoch
        #     )
        # else:
        #     reviving_neurons = self.net.get_reviving_neurons(
        #         layer_name, self.active_task
        #     )

        # if current_epoch is not None and prev_epoch is not None:
        reviving_neurons = self.net.find_persistently_reviving_neurons(
            active_task=self.active_task, layer_name=layer_name, n_epochs=50
        )["persistently_reviving_neurons"]

        if not reviving_neurons:
            return

        print(f"Reviving neurons in {layer_name}: {reviving_neurons}")
        print(
            f"Re-initializing {len(reviving_neurons)} reviving neurons in {layer_name}"
        )

        if hasattr(layer, "weight"):
            # For convolutional layers
            if len(layer.weight.shape) == 4:
                for neuron_idx in reviving_neurons:
                    # Initialize incoming weights using Kaiming normal
                    torch.nn.init.kaiming_normal_(
                        layer.weight[neuron_idx : neuron_idx + 1],
                        mode="fan_in",
                        nonlinearity="relu",
                    )

                    # Find next layer for outgoing connections
                    next_layer_name = self._get_next_layer_name(layer_name)
                    if next_layer_name:
                        next_layer = getattr(self.net, next_layer_name)
                        if (
                            hasattr(next_layer, "weight")
                            and len(next_layer.weight.shape) == 4
                        ):
                            # For conv layers, initialize weights that use this neuron's output as input
                            torch.nn.init.kaiming_normal_(
                                next_layer.weight[:, neuron_idx : neuron_idx + 1],
                                mode="fan_out",
                                nonlinearity="relu",
                            )

            # For fully connected layers
            elif len(layer.weight.shape) == 2:
                for neuron_idx in reviving_neurons:
                    # Initialize incoming weights using Kaiming normal
                    torch.nn.init.kaiming_normal_(
                        layer.weight[neuron_idx : neuron_idx + 1],
                        mode="fan_in",
                        nonlinearity="relu",
                    )

                    # Find next layer for outgoing connections
                    next_layer_name = self._get_next_layer_name(layer_name)
                    if next_layer_name:
                        next_layer = getattr(self.net, next_layer_name)
                        if (
                            hasattr(next_layer, "weight")
                            and len(next_layer.weight.shape) == 2
                        ):
                            # Initialize outgoing weights for this neuron
                            torch.nn.init.kaiming_normal_(
                                next_layer.weight[:, neuron_idx : neuron_idx + 1],
                                mode="fan_out",
                                nonlinearity="relu",
                            )

        # Initialize biases to zero (standard practice with Kaiming init)
        if hasattr(layer, "bias") and layer.bias is not None:
            for neuron_idx in reviving_neurons:
                layer.bias.data[neuron_idx] = 0

        # Handle BatchNorm parameters for the corresponding layer
        self._reinitialize_batchnorm_for_neurons(layer_name, reviving_neurons)

    def _reinitialize_batchnorm_for_neurons(self, layer_name, neuron_indices):
        """
        Reinitializes batch normalization parameters for specific neurons.

        Args:
            layer_name: Name of the layer (like "layer1", "layer2", etc.)
            neuron_indices: List of neuron indices to reinitialize
        """
        if not neuron_indices:
            return

        # Handle main network BatchNorm (for conv1)
        if layer_name == "conv1" and hasattr(self.net, "bn1"):
            bn_layer = self.net.bn1
            for idx in neuron_indices:
                if idx < bn_layer.weight.size(0):
                    # Reset gamma (weight) to 1
                    bn_layer.weight.data[idx] = 1.0
                    # Reset beta (bias) to 0
                    if bn_layer.bias is not None:
                        bn_layer.bias.data[idx] = 0.0
                    # Reset running mean to 0 and running var to 1
                    if hasattr(bn_layer, "running_mean") and hasattr(
                        bn_layer, "running_var"
                    ):
                        bn_layer.running_mean.data[idx] = 0.0
                        bn_layer.running_var.data[idx] = 1.0

        # Handle layer's BatchNorm (for ResNet blocks)
        elif layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_seq = getattr(self.net, layer_name)
            for block_idx, block in enumerate(layer_seq):
                # For BasicBlock architecture, check bn1 and bn2
                if hasattr(block, "bn1"):
                    bn_layer = block.bn1
                    for idx in neuron_indices:
                        if idx < bn_layer.weight.size(0):
                            # Reset gamma (weight) to 1
                            bn_layer.weight.data[idx] = 1.0
                            # Reset beta (bias) to 0
                            if bn_layer.bias is not None:
                                bn_layer.bias.data[idx] = 0.0
                            # Reset running mean to 0 and running var to 1
                            if hasattr(bn_layer, "running_mean") and hasattr(
                                bn_layer, "running_var"
                            ):
                                bn_layer.running_mean.data[idx] = 0.0
                                bn_layer.running_var.data[idx] = 1.0

                # In case there's a second BatchNorm (as in BasicBlock)
                if hasattr(block, "bn2"):
                    # For bn2, we reinitialize all parameters since it's affected by
                    # the revived neurons in the previous layer
                    bn_layer = block.bn2

                    # We don't reset bn2 per neuron since its input depends on all
                    # neurons from the previous layer. Instead, we could apply a similar
                    # reset logic to output channels that are affected by input channels
                    # that were revived, but this requires more complex tracking.

    def _apply_pruning_hooks(self, layer_name, permanently_dead_neurons):
        """
        Apply hooks to force zero activations for pruned neurons after each forward pass.
        """
        # Special handling for conv1 which is a single layer
        if layer_name == "conv1":
            if not hasattr(self.net, layer_name):
                return

            layer = getattr(self.net, layer_name)

            def make_hook(neurons_to_prune):
                def hook(module, input, output):
                    for idx in neurons_to_prune:
                        if idx < output.size(1):
                            output[:, idx] = 0.0
                    return output

                return hook

            if hasattr(layer, "_pruning_hook"):
                layer._pruning_hook.remove()
            layer._pruning_hook = layer.register_forward_hook(
                make_hook(permanently_dead_neurons)
            )
            return

        # Original code for layer1-4
        layer_seq = getattr(self.net, layer_name)

        def make_hook(neurons_to_prune):
            def hook(module, input, output):
                for idx in neurons_to_prune:
                    if idx < output.size(1):
                        output[:, idx] = 0.0
                return output

            return hook

        for block in layer_seq:
            if hasattr(block, "conv1"):
                if hasattr(block, "_pruning_hook"):
                    block._pruning_hook.remove()
                block._pruning_hook = block.register_forward_hook(
                    make_hook(permanently_dead_neurons)
                )

    def _test_pruning_effectiveness(self, layer_name, permanently_dead_neurons):
        """
        Test if pruned neurons are producing zero activations.
        """
        print(f"\nTesting effectiveness of pruning for {layer_name}...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 32, 32).to(self.device)
            try:
                _ = self.net(test_input)
                activations = self.net.activations[layer_name]
                effective = True
                for neuron_idx in permanently_dead_neurons:
                    if neuron_idx >= activations.size(1):
                        print(
                            f"  ⚠️ Neuron index {neuron_idx} out of bounds (max: {activations.size(1)-1})"
                        )
                        continue
                    activation_sum = activations[:, neuron_idx].abs().sum().item()
                    print(
                        f"  Neuron {neuron_idx}: activation sum = {activation_sum:.6e}"
                    )
                    if activation_sum > 1e-6:
                        effective = False
                if effective:
                    print(f"✓ Pruning effective for {layer_name}")
                else:
                    print(f"✗ Pruning ineffective for {layer_name} - debug needed")
            except Exception as e:
                print(f"❌ Error during pruning effectiveness test: {str(e)}")
                return False
        return effective

    def get_current_pruning_rate(self):
        """
        Compute the current pruning rate based on task number.
        You can use a linear or exponential schedule. Here we use a simple linear increase.
        """
        # Example: Increase pruning rate by 5% every task
        return min(
            self.base_pruning_rate + 0.05 * self.active_task / 2, 0.2
        )  # cap at 50%

    def prune_dead_neurons(self, layer_name):
        """
        Modified prune_dead_neurons that prunes a fraction of dead neurons based on current pruning rate.
        """
        if self.active_task == 0:
            return

        # Special handling for conv1 which is a single Conv2d layer, not a sequence
        if layer_name == "conv1":
            if not hasattr(self.net, layer_name):
                return

            result = self.net.find_permanently_dead_neurons(layer_name)
            permanently_dead_neurons = result["permanent_dead_neurons"]
            # permanently_dead_neurons = (
            #     self.net.find_persistently_dead_neurons_accross_tasks(
            #         active_task=self.active_task,
            #         layer_name=layer_name,
            #         n_epochs=50,
            #         n_tasks=2,
            #     )
            # )["consecutively_dead_neurons"]

            if not permanently_dead_neurons:
                return

            current_rate = self.get_current_pruning_rate()
            num_to_prune = max(1, int(len(permanently_dead_neurons) * current_rate))
            neurons_to_prune = sorted(permanently_dead_neurons)[:num_to_prune]

            print(
                f"Pruning {len(neurons_to_prune)} neurons (of {len(permanently_dead_neurons)} detected) in {layer_name} at rate {current_rate*100:.1f}%"
            )

            if not hasattr(self, "pruned_neurons_count"):
                self.pruned_neurons_count = {}
            self.pruned_neurons_count[layer_name] = self.pruned_neurons_count.get(
                layer_name, 0
            ) + len(neurons_to_prune)

            # Get the conv1 layer directly
            conv_layer = getattr(self.net, layer_name)

            # Add weight mask if it doesn't exist
            if not hasattr(conv_layer, "weight_mask"):
                conv_layer.register_buffer(
                    "weight_mask", torch.ones_like(conv_layer.weight)
                )

            weight_mask_size = conv_layer.weight_mask.size(0)
            for neuron_idx in neurons_to_prune:
                if neuron_idx < weight_mask_size:
                    conv_layer.weight_mask[neuron_idx] = 0.0
                    conv_layer.weight.data[neuron_idx] = 0.0
                else:
                    print(
                        f"Warning: Neuron index {neuron_idx} out of bounds (max: {weight_mask_size-1})"
                    )

            # Handle bias if it exists
            if hasattr(conv_layer, "bias") and conv_layer.bias is not None:
                if not hasattr(conv_layer, "bias_mask"):
                    conv_layer.register_buffer(
                        "bias_mask", torch.ones_like(conv_layer.bias)
                    )
                bias_mask_size = conv_layer.bias_mask.size(0)
                for neuron_idx in neurons_to_prune:
                    if neuron_idx < bias_mask_size:
                        conv_layer.bias_mask[neuron_idx] = 0.0
                        conv_layer.bias.data[neuron_idx] = 0.0
                    else:
                        print(
                            f"Warning: Neuron index {neuron_idx} out of bounds for bias (max: {bias_mask_size-1})"
                        )

            # Handle applying hooks for conv1
            def make_hook(neurons_to_prune):
                def hook(module, input, output):
                    for idx in neurons_to_prune:
                        if idx < output.size(1):
                            output[:, idx] = 0.0
                    return output

                return hook

            if hasattr(conv_layer, "_pruning_hook"):
                conv_layer._pruning_hook.remove()
            conv_layer._pruning_hook = conv_layer.register_forward_hook(
                make_hook(neurons_to_prune)
            )

            # Test effectiveness
            self._test_pruning_effectiveness(layer_name, neurons_to_prune)

            return

        # Original code for layer1-4 which are sequences of blocks
        layer_seq = getattr(self.net, layer_name)
        result = self.net.find_permanently_dead_neurons(layer_name)
        permanently_dead_neurons = result["permanent_dead_neurons"]
        # permanently_dead_neurons = (
        #     self.net.find_persistently_dead_neurons_accross_tasks(
        #         active_task=self.active_task,
        #         layer_name=layer_name,
        #         n_epochs=50,
        #         n_tasks=2,
        #     )
        # )["consecutively_dead_neurons"]

        if not permanently_dead_neurons:
            return

        current_rate = self.get_current_pruning_rate()
        num_to_prune = max(1, int(len(permanently_dead_neurons) * current_rate))
        neurons_to_prune = sorted(permanently_dead_neurons)[:num_to_prune]

        print(
            f"Pruning {len(neurons_to_prune)} neurons (of {len(permanently_dead_neurons)} detected) in {layer_name} at rate {current_rate*100:.1f}%"
        )

        if not hasattr(self, "pruned_neurons_count"):
            self.pruned_neurons_count = {}
        self.pruned_neurons_count[layer_name] = self.pruned_neurons_count.get(
            layer_name, 0
        ) + len(neurons_to_prune)

        for block in layer_seq:
            if hasattr(block, "conv1"):
                if not hasattr(block.conv1, "weight_mask"):
                    block.conv1.register_buffer(
                        "weight_mask", torch.ones_like(block.conv1.weight)
                    )
                weight_mask_size = block.conv1.weight_mask.size(0)
                for neuron_idx in neurons_to_prune:
                    if neuron_idx < weight_mask_size:
                        block.conv1.weight_mask[neuron_idx] = 0.0
                        block.conv1.weight.data[neuron_idx] = 0.0
                    else:
                        print(
                            f"Warning: Neuron index {neuron_idx} out of bounds (max: {weight_mask_size-1})"
                        )
                if hasattr(block.conv1, "bias") and block.conv1.bias is not None:
                    if not hasattr(block.conv1, "bias_mask"):
                        block.conv1.register_buffer(
                            "bias_mask", torch.ones_like(block.conv1.bias)
                        )
                    bias_mask_size = block.conv1.bias_mask.size(0)
                    for neuron_idx in neurons_to_prune:
                        if neuron_idx < bias_mask_size:
                            block.conv1.bias_mask[neuron_idx] = 0.0
                            block.conv1.bias.data[neuron_idx] = 0.0
                        else:
                            print(
                                f"Warning: Neuron index {neuron_idx} out of bounds for bias (max: {bias_mask_size-1})"
                            )
            if hasattr(block, "conv2"):
                if not hasattr(block.conv2, "weight_mask"):
                    block.conv2.register_buffer(
                        "weight_mask", torch.ones_like(block.conv2.weight)
                    )
                weight_mask_size = block.conv2.weight_mask.size(1)
                for neuron_idx in neurons_to_prune:
                    if neuron_idx < weight_mask_size:
                        block.conv2.weight_mask[:, neuron_idx] = 0.0
                        block.conv2.weight.data[:, neuron_idx] = 0.0
                    else:
                        print(
                            f"Warning: Neuron index {neuron_idx} out of bounds for conv2 input (max: {weight_mask_size-1})"
                        )

        # Apply hooks and test effectiveness as before
        self._apply_pruning_hooks(layer_name, neurons_to_prune)
        self._test_pruning_effectiveness(layer_name, neurons_to_prune)

    def verify_pruning_effectiveness(self):
        """
        Verifies that pruned neurons are removed from computation by counting parameters and checking activations.
        """
        results = {}
        layers_to_check = ["layer1", "layer2", "layer3", "layer4"]
        total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        results["total_params"] = total_params
        pruned_params = 0
        pruned_neurons_by_layer = {}

        for layer_name in layers_to_check:
            if not hasattr(self.net, layer_name):
                continue
            layer_seq = getattr(self.net, layer_name)
            for block in layer_seq:
                if hasattr(block, "conv1") and hasattr(block.conv1, "weight_mask"):
                    pruned_params += (block.conv1.weight_mask == 0).sum().item()
                    if (
                        hasattr(block.conv1, "bias_mask")
                        and block.conv1.bias is not None
                    ):
                        pruned_params += (block.conv1.bias_mask == 0).sum().item()
                if hasattr(block, "conv2") and hasattr(block.conv2, "weight_mask"):
                    pruned_params += (block.conv2.weight_mask == 0).sum().item()
                    if (
                        hasattr(block.conv2, "bias_mask")
                        and block.conv2.bias is not None
                    ):
                        pruned_params += (block.conv2.bias_mask == 0).sum().item()

        results["pruned_params"] = pruned_params
        results["pruned_percentage"] = (
            (pruned_params / total_params * 100) if total_params > 0 else 0
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
            _ = self.net(dummy_input)
        test_effectiveness = {}
        for layer_name in layers_to_check:
            if not hasattr(self.net, layer_name):
                continue
            if layer_name in self.net.activations:
                pruned_neurons = set()
                layer_seq = getattr(self.net, layer_name)
                for block in layer_seq:
                    if hasattr(block, "conv1") and hasattr(block.conv1, "weight_mask"):
                        pruned_list = torch.where(
                            block.conv1.weight_mask.sum(dim=(1, 2, 3)) == 0
                        )[0]
                        pruned_neurons.update(pruned_list.tolist())
                if not pruned_neurons:
                    continue
                pruned_neurons = list(pruned_neurons)
                pruned_neurons_by_layer[layer_name] = len(pruned_neurons)
                test_effectiveness[layer_name] = {
                    "pruned_count": len(pruned_neurons),
                    "activation_zero": False,
                }
                activations = self.net.activations[layer_name]
                if len(pruned_neurons) > 0:
                    pruned_activations = activations[:, pruned_neurons]
                    abs_sum = pruned_activations.abs().sum().item()
                    test_effectiveness[layer_name]["activation_abs_sum"] = abs_sum
                    if abs_sum < 1e-6:
                        test_effectiveness[layer_name]["activation_zero"] = True

        results["pruning_effectiveness"] = test_effectiveness
        self.pruned_neurons_count = pruned_neurons_by_layer

        print("\n=== PRUNING VERIFICATION ===")
        print(f"Total parameters: {total_params}")
        print(
            f"Pruned parameters: {pruned_params} ({results['pruned_percentage']:.2f}%)"
        )
        print("\nEffectiveness check:")
        for layer_name, metrics in test_effectiveness.items():
            status = "✓ EFFECTIVE" if metrics["activation_zero"] else "✗ INEFFECTIVE"
            if not metrics["activation_zero"] and "activation_abs_sum" in metrics:
                status += f" (sum={metrics['activation_abs_sum']:.6e})"
            print(
                f"  {layer_name}: {metrics['pruned_count']} neurons pruned - {status}"
            )

        ineffective_layers = [
            layer
            for layer, metrics in test_effectiveness.items()
            if not metrics["activation_zero"]
        ]
        if ineffective_layers:
            print("\nDebug information for ineffective layers:")
            for layer_name in ineffective_layers:
                print(f"  {layer_name}:")
                layer_seq = getattr(self.net, layer_name)
                for block_idx, block in enumerate(layer_seq):
                    if hasattr(block, "conv1") and hasattr(block.conv1, "weight_mask"):
                        zeros = (block.conv1.weight_mask == 0).sum().item()
                        total = block.conv1.weight_mask.numel()
                        print(
                            f"    Block {block_idx} conv1: {zeros}/{total} zeros in mask"
                        )
                    if hasattr(block, "conv2") and hasattr(block.conv2, "weight_mask"):
                        zeros = (block.conv2.weight_mask == 0).sum().item()
                        total = block.conv2.weight_mask.numel()
                        print(
                            f"    Block {block_idx} conv2: {zeros}/{total} zeros in mask"
                        )
        self.fix_dimension_mismatches(debug=False)
        return results

    def fix_dimension_mismatches(self, debug=False):
        """
        Consolidated function to check and fix channel mismatches throughout the network.
        Revised so that the shortcut connections expect the block's original input channels.
        """
        print("\n=== Fixing Dimension Mismatches ===")
        # Start with the first conv layer
        prev_channels = self.net.conv1.weight.size(0)
        print(f"conv1: in={self.net.conv1.weight.size(1)}, out={prev_channels}")

        # Iterate over the main blocks in each layer
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if not hasattr(self.net, layer_name) or isinstance(
                getattr(self.net, layer_name), nn.Identity
            ):
                print(f"{layer_name}: Skipping (not present or Identity)")
                continue

            layer_seq = getattr(self.net, layer_name)
            for block_idx, block in enumerate(layer_seq):
                # Save the block's input channel count before modifying the block
                block_input_channels = prev_channels

                # --- Fix conv1: its input must match the block's input channels ---
                if hasattr(block, "conv1"):
                    in_channels = block.conv1.weight.size(1)
                    out_channels = block.conv1.weight.size(0)
                    if in_channels != block_input_channels:
                        print(
                            f"Fixing {layer_name}[{block_idx}].conv1: in={in_channels}, expected={block_input_channels}"
                        )
                        old_weight = block.conv1.weight.data
                        new_weight = torch.zeros(
                            out_channels,
                            block_input_channels,
                            old_weight.size(2),
                            old_weight.size(3),
                            device=old_weight.device,
                        )
                        min_channels = min(in_channels, block_input_channels)
                        new_weight[:, :min_channels] = old_weight[:, :min_channels]
                        block.conv1.weight = nn.Parameter(new_weight)
                        if hasattr(block.conv1, "weight_mask"):
                            old_mask = block.conv1.weight_mask
                            new_mask = torch.ones(
                                out_channels,
                                block_input_channels,
                                old_mask.size(2),
                                old_mask.size(3),
                                device=old_mask.device,
                            )
                            new_mask[:, :min_channels] = old_mask[:, :min_channels]
                            block.conv1.weight_mask = new_mask
                    conv1_out = block.conv1.weight.size(0)
                else:
                    conv1_out = block_input_channels

                # --- Fix shortcut: expected input should be the block's original input channels ---
                if hasattr(block, "shortcut") and len(block.shortcut) > 0:
                    for i, module in enumerate(block.shortcut):
                        if isinstance(module, nn.Conv2d):
                            sc_in = module.weight.size(1)
                            if sc_in != block_input_channels:
                                print(
                                    f"Fixing {layer_name}[{block_idx}].shortcut[{i}]: in={sc_in}, expected={block_input_channels}"
                                )
                                old_weight = module.weight.data
                                new_weight = torch.zeros(
                                    module.weight.size(0),
                                    block_input_channels,
                                    old_weight.size(2),
                                    old_weight.size(3),
                                    device=old_weight.device,
                                )
                                min_channels = min(sc_in, block_input_channels)
                                new_weight[:, :min_channels] = old_weight[
                                    :, :min_channels
                                ]
                                module.weight = nn.Parameter(new_weight)

                # --- Fix conv2: its input should match conv1's output ---
                if hasattr(block, "conv2"):
                    in_channels = block.conv2.weight.size(1)
                    out_channels = block.conv2.weight.size(0)
                    if in_channels != conv1_out:
                        print(
                            f"Fixing {layer_name}[{block_idx}].conv2: in={in_channels}, expected={conv1_out}"
                        )
                        old_weight = block.conv2.weight.data
                        new_weight = torch.zeros(
                            out_channels,
                            conv1_out,
                            old_weight.size(2),
                            old_weight.size(3),
                            device=old_weight.device,
                        )
                        min_channels = min(in_channels, conv1_out)
                        new_weight[:, :min_channels] = old_weight[:, :min_channels]
                        block.conv2.weight = nn.Parameter(new_weight)
                        if hasattr(block.conv2, "weight_mask"):
                            old_mask = block.conv2.weight_mask
                            new_mask = torch.ones(
                                out_channels,
                                conv1_out,
                                old_mask.size(2),
                                old_mask.size(3),
                                device=old_mask.device,
                            )
                            new_mask[:, :min_channels] = old_mask[:, :min_channels]
                            block.conv2.weight_mask = new_mask

                # For the next block, update prev_channels as the output of conv2 if it exists; otherwise use conv1's output.
                prev_channels = (
                    block.conv2.weight.size(0) if hasattr(block, "conv2") else conv1_out
                )

        if debug:
            try:
                test_input = torch.randn(2, 3, 32, 32, device=self.device)
                output = self.net(test_input)
                print(f"✓ Final forward pass successful. Output shape: {output.shape}")
            except Exception as e:
                print(f"✗ Final forward pass failed: {str(e)}")
        return True

    def _transfer_weights(self, old_net, new_net, pruning_config):
        """
        Transfer weights from old network to new pruned network,
        including BatchNorm parameters.
        """
        # Transfer conv1 weights and bn1 parameters
        new_net.conv1.weight.data = old_net.conv1.weight.data.clone()
        if old_net.conv1.bias is not None:
            new_net.conv1.bias.data = old_net.conv1.bias.data.clone()

        # Transfer bn1 parameters
        if hasattr(old_net, "bn1") and hasattr(new_net, "bn1"):
            new_net.bn1.weight.data = old_net.bn1.weight.data.clone()
            new_net.bn1.bias.data = old_net.bn1.bias.data.clone()
            new_net.bn1.running_mean.data = old_net.bn1.running_mean.data.clone()
            new_net.bn1.running_var.data = old_net.bn1.running_var.data.clone()

        prev_out_dim = new_net.conv1.weight.size(0)
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if isinstance(getattr(new_net, layer_name), nn.Identity):
                continue

            if layer_name not in pruning_config:
                # For layers without pruning, transfer all weights directly
                old_layer = getattr(old_net, layer_name)
                new_layer = getattr(new_net, layer_name)
                for old_block, new_block in zip(old_layer, new_layer):
                    # Transfer conv1 weights and bn1 parameters
                    if hasattr(old_block, "conv1"):
                        new_block.conv1.weight.data = (
                            old_block.conv1.weight.data.clone()
                        )
                        if (
                            hasattr(old_block.conv1, "bias")
                            and old_block.conv1.bias is not None
                        ):
                            new_block.conv1.bias.data = (
                                old_block.conv1.bias.data.clone()
                            )

                        # Transfer bn1 parameters
                        new_block.bn1.weight.data = old_block.bn1.weight.data.clone()
                        new_block.bn1.bias.data = old_block.bn1.bias.data.clone()
                        new_block.bn1.running_mean.data = (
                            old_block.bn1.running_mean.data.clone()
                        )
                        new_block.bn1.running_var.data = (
                            old_block.bn1.running_var.data.clone()
                        )

                    # Transfer conv2 weights and bn2 parameters
                    if hasattr(old_block, "conv2"):
                        new_block.conv2.weight.data = (
                            old_block.conv2.weight.data.clone()
                        )
                        if (
                            hasattr(old_block.conv2, "bias")
                            and old_block.conv2.bias is not None
                        ):
                            new_block.conv2.bias.data = (
                                old_block.conv2.bias.data.clone()
                            )

                        # Transfer bn2 parameters
                        new_block.bn2.weight.data = old_block.bn2.weight.data.clone()
                        new_block.bn2.bias.data = old_block.bn2.bias.data.clone()
                        new_block.bn2.running_mean.data = (
                            old_block.bn2.running_mean.data.clone()
                        )
                        new_block.bn2.running_var.data = (
                            old_block.bn2.running_var.data.clone()
                        )

                    # Transfer shortcut weights if it exists
                    if len(old_block.shortcut) > 0 and len(new_block.shortcut) > 0:
                        for i, (old_module, new_module) in enumerate(
                            zip(old_block.shortcut, new_block.shortcut)
                        ):
                            if isinstance(old_module, nn.Conv2d):
                                new_module.weight.data = old_module.weight.data.clone()
                            elif isinstance(old_module, nn.BatchNorm2d):
                                new_module.weight.data = old_module.weight.data.clone()
                                new_module.bias.data = old_module.bias.data.clone()
                                new_module.running_mean.data = (
                                    old_module.running_mean.data.clone()
                                )
                                new_module.running_var.data = (
                                    old_module.running_var.data.clone()
                                )

                prev_out_dim = getattr(new_net, layer_name)[0].conv1.weight.size(0)
            else:
                # For pruned layers, transfer weights selectively
                old_layer = getattr(old_net, layer_name)
                new_layer = getattr(new_net, layer_name)
                keep_indices = pruning_config[layer_name]["keep_indices"]
                keep_tensor = torch.tensor(
                    keep_indices, device=old_net.conv1.weight.device
                )

                for old_block, new_block in zip(old_layer, new_layer):
                    # Transfer conv1 weights and bn1 parameters
                    if hasattr(old_block, "conv1"):
                        # Transfer pruned conv1 weights
                        new_block.conv1.weight.data = (
                            old_block.conv1.weight.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )
                        if (
                            hasattr(old_block.conv1, "bias")
                            and old_block.conv1.bias is not None
                        ):
                            new_block.conv1.bias.data = (
                                old_block.conv1.bias.data.index_select(
                                    0, keep_tensor
                                ).clone()
                            )

                        # Transfer pruned bn1 parameters
                        new_block.bn1.weight.data = (
                            old_block.bn1.weight.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )
                        new_block.bn1.bias.data = old_block.bn1.bias.data.index_select(
                            0, keep_tensor
                        ).clone()
                        new_block.bn1.running_mean.data = (
                            old_block.bn1.running_mean.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )
                        new_block.bn1.running_var.data = (
                            old_block.bn1.running_var.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )

                    # Transfer conv2 weights and bn2 parameters
                    if hasattr(old_block, "conv2"):
                        # Transfer pruned conv2 weights (both input and output channels)
                        conv2_weight = old_block.conv2.weight.data.index_select(
                            0, keep_tensor
                        )
                        conv2_weight = conv2_weight.index_select(1, keep_tensor)
                        new_block.conv2.weight.data = conv2_weight.clone()

                        if (
                            hasattr(old_block.conv2, "bias")
                            and old_block.conv2.bias is not None
                        ):
                            new_block.conv2.bias.data = (
                                old_block.conv2.bias.data.index_select(
                                    0, keep_tensor
                                ).clone()
                            )

                        # Transfer pruned bn2 parameters
                        new_block.bn2.weight.data = (
                            old_block.bn2.weight.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )
                        new_block.bn2.bias.data = old_block.bn2.bias.data.index_select(
                            0, keep_tensor
                        ).clone()
                        new_block.bn2.running_mean.data = (
                            old_block.bn2.running_mean.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )
                        new_block.bn2.running_var.data = (
                            old_block.bn2.running_var.data.index_select(
                                0, keep_tensor
                            ).clone()
                        )

                    # Transfer shortcut weights if it exists
                    if len(old_block.shortcut) > 0 and len(new_block.shortcut) > 0:
                        for i, (old_module, new_module) in enumerate(
                            zip(old_block.shortcut, new_block.shortcut)
                        ):
                            if isinstance(old_module, nn.Conv2d) and isinstance(
                                new_module, nn.Conv2d
                            ):
                                # For conv in shortcut, we need to keep output channels
                                shortcut_weight = old_module.weight.data.index_select(
                                    0, keep_tensor
                                )
                                min_in_channels = min(
                                    shortcut_weight.size(1), new_module.weight.size(1)
                                )
                                new_module.weight.data[:, :min_in_channels] = (
                                    shortcut_weight[:, :min_in_channels]
                                )
                            elif isinstance(old_module, nn.BatchNorm2d) and isinstance(
                                new_module, nn.BatchNorm2d
                            ):
                                # For BN in shortcut, transfer pruned parameters
                                new_module.weight.data = (
                                    old_module.weight.data.index_select(
                                        0, keep_tensor
                                    ).clone()
                                )
                                new_module.bias.data = (
                                    old_module.bias.data.index_select(
                                        0, keep_tensor
                                    ).clone()
                                )
                                new_module.running_mean.data = (
                                    old_module.running_mean.data.index_select(
                                        0, keep_tensor
                                    ).clone()
                                )
                                new_module.running_var.data = (
                                    old_module.running_var.data.index_select(
                                        0, keep_tensor
                                    ).clone()
                                )

                prev_out_dim = getattr(new_net, layer_name)[0].conv1.weight.size(0)

        # Transfer classifier weights
        final_feature_dim = prev_out_dim
        new_classifier = nn.Linear(
            final_feature_dim, new_net.classifier.out_features
        ).to(new_net.classifier.weight.device)
        min_features = min(old_net.classifier.in_features, final_feature_dim)
        new_classifier.weight.data[:, :min_features] = old_net.classifier.weight.data[
            :, :min_features
        ].clone()
        new_classifier.bias.data = old_net.classifier.bias.data.clone()
        new_net.classifier = new_classifier

        print("\n✓ Weight transfer complete (including BatchNorm parameters).")
        return new_net

    def update_classifier_after_pruning(self, new_net):
        """
        Update the classifier to match the feature dimensions of the backbone.
        """
        with torch.no_grad():
            try:
                x = new_net.conv1(torch.randn(2, 3, 32, 32, device=self.device))
                for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                    if not isinstance(getattr(new_net, layer_name), nn.Identity):
                        x = getattr(new_net, layer_name)(x)
                features = torch.flatten(x, 1)
            except Exception as e:
                print(f"Error in feature extraction: {e}")
                return new_net

        actual_dim = features.size(1)
        old_classifier = self.net.classifier
        out_features = old_classifier.out_features
        new_classifier = nn.Linear(actual_dim, out_features).to(self.device)
        min_features = min(old_classifier.in_features, actual_dim)
        new_classifier.weight.data[:, :min_features] = old_classifier.weight.data[
            :, :min_features
        ].clone()
        new_classifier.bias.data = old_classifier.bias.data.clone()
        new_net.classifier = new_classifier

        print(
            f"✓ Updated classifier: in_features {actual_dim} -> out_features {out_features}"
        )
        return new_net

    def _build_custom_resnet(self, pruning_config):
        """
        Builds a custom ResNet with specific channel counts based on pruning configuration.
        """
        from backbone.ResNetBlock import ResNet, BasicBlock
        import copy

        class CustomResNet(ResNet):
            def __init__(self, block, num_blocks, num_classes, channels_per_layer):
                super().__init__(
                    block=block,
                    num_blocks=num_blocks,
                    num_classes=num_classes,
                    nf=channels_per_layer["conv1"],
                )
                self.in_planes = channels_per_layer["conv1"]
                self.conv1 = nn.Conv2d(
                    3,
                    channels_per_layer["conv1"],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.layer1 = self._make_layer(
                    block, channels_per_layer["layer1"], num_blocks[0], stride=1
                )
                self.in_planes = channels_per_layer["layer1"]
                self.layer2 = self._make_layer(
                    block, channels_per_layer["layer2"], num_blocks[1], stride=2
                )
                self.in_planes = channels_per_layer["layer2"]
                self.layer3 = self._make_layer(
                    block, channels_per_layer["layer3"], num_blocks[2], stride=2
                )
                self.in_planes = channels_per_layer["layer3"]
                self.layer4 = self._make_layer(
                    block, channels_per_layer["layer4"], num_blocks[3], stride=2
                )
                self.classifier = nn.Linear(channels_per_layer["layer4"], num_classes)
                self.channels_per_layer = channels_per_layer

                print("\nActual layer dimensions after construction:")
                print(f"  conv1 out: {self.conv1.weight.size(0)}")
                for i, layer_name in enumerate(
                    ["layer1", "layer2", "layer3", "layer4"]
                ):
                    layer = getattr(self, layer_name)
                    if isinstance(layer, nn.Sequential) and len(layer) > 0:
                        in_ch = layer[0].conv1.weight.size(1)
                        out_ch = layer[0].conv1.weight.size(0)
                        print(f"  {layer_name}: in={in_ch}, out={out_ch}")

        orig_channels = {}
        orig_channels["conv1"] = self.net.conv1.weight.size(0)
        orig_channels["layer1"] = self.net.layer1[0].conv1.weight.size(0)
        orig_channels["layer2"] = self.net.layer2[0].conv1.weight.size(0)
        orig_channels["layer3"] = self.net.layer3[0].conv1.weight.size(0)
        orig_channels["layer4"] = (
            self.net.layer4[0].conv1.weight.size(0)
            if not isinstance(self.net.layer4, nn.Identity)
            else 0
        )

        new_channels = orig_channels.copy()
        for layer_name, config in pruning_config.items():
            new_channels[layer_name] = len(config["keep_indices"])

        print("\nNetwork channel configuration:")
        for name, count in new_channels.items():
            print(f"  {name}: {count} channels")

        custom_net = CustomResNet(
            BasicBlock, [2, 2, 2, 2], self.num_classes, channels_per_layer=new_channels
        )
        return custom_net

    def fix_classifier_dimensions(self):
        """
        Fix the classifier layer to match the feature dimensions coming from the network.
        """
        print("\nChecking classifier dimensions...")
        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32, device=self.device)
            try:
                if hasattr(self.net, "features") and callable(
                    getattr(self.net, "features")
                ):
                    features = self.net.features(x)
                else:
                    x = self.net.conv1(x)
                    if hasattr(self.net, "bn1"):
                        x = self.net.bn1(x)
                    x = self.net.layer1(x)
                    x = self.net.layer2(x)
                    x = self.net.layer3(x)
                    if not isinstance(self.net.layer4, nn.Identity):
                        x = self.net.layer4(x)
                    if hasattr(self.net, "avgpool"):
                        x = self.net.avgpool(x)
                    features = torch.flatten(x, 1)
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                return False

        actual_features = features.size(1)
        expected_features = self.net.classifier.weight.size(1)
        print(f"Actual feature dimension: {actual_features}")
        print(f"Current classifier input dimension: {expected_features}")
        if actual_features != expected_features:
            print(
                f"Fixing classifier mismatch: {expected_features} -> {actual_features}"
            )
            old_classifier = self.net.classifier
            new_classifier = nn.Linear(actual_features, old_classifier.out_features).to(
                old_classifier.weight.device
            )
            min_features = min(old_classifier.in_features, actual_features)
            new_classifier.weight.data[:, :min_features] = old_classifier.weight.data[
                :, :min_features
            ].clone()
            new_classifier.bias.data = old_classifier.bias.data.clone()
            self.net.classifier = new_classifier
            print(f"Classifier fixed. New shape: {self.net.classifier.weight.shape}")
            return True
        print("Classifier dimensions are correct")
        return False

    def update_dead_neuron_tracking_after_pruning(self):
        """
        Update dead neuron tracking after network has been rebuilt.
        """
        print("\nUpdating dead neuron tracking for new network structure...")
        if hasattr(self.net, "dead_neuron_history"):
            for layer_name in list(self.net.dead_neuron_history.keys()):
                if hasattr(self.net, layer_name) and not isinstance(
                    getattr(self.net, layer_name), nn.Identity
                ):
                    try:
                        layer = getattr(self.net, layer_name)[0]
                        new_history = {
                            neuron_idx: {}
                            for neuron_idx in range(layer.conv1.weight.size(0))
                        }
                        self.net.dead_neuron_history[layer_name] = new_history
                        if (
                            hasattr(self.net, "permanently_dead_neurons")
                            and layer_name in self.net.permanently_dead_neurons
                        ):
                            self.net.permanently_dead_neurons[layer_name] = []
                    except Exception as e:
                        print(f"  Error updating {layer_name}: {str(e)}")
                        if layer_name in self.net.dead_neuron_history:
                            del self.net.dead_neuron_history[layer_name]
                else:
                    print(
                        f"  {layer_name}: Layer not found or is Identity, removing tracking"
                    )
                    if layer_name in self.net.dead_neuron_history:
                        del self.net.dead_neuron_history[layer_name]
        if not hasattr(self.net, "activations"):
            self.net.activations = {}
        if hasattr(self.net, "overlap_ratios"):
            self.net.overlap_ratios = {}
        self.pruned_neurons_count = {}
        print("Dead neuron tracking updated")
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 3, 32, 32, device=self.device)
                _ = self.net(test_input)
                print("✓ Forward pass verification successful")
        except Exception as e:
            print(f"⚠️ Forward pass verification failed: {str(e)}")

    def rebuild_network_with_pruning(self):
        """
        Rebuilds the network by structurally removing pruned neurons.
        """
        if not hasattr(self, "pruned_neurons_count") or not self.pruned_neurons_count:
            print("No pruning information available. Cannot rebuild network.")
            return False

        print("\n=== Rebuilding Network with Structural Pruning ===")
        old_net = self.net
        original_params = sum(p.numel() for p in old_net.parameters())
        pruning_config = {}
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if (
                layer_name in self.pruned_neurons_count
                and self.pruned_neurons_count[layer_name] > 0
            ):
                first_block = getattr(old_net, layer_name)[0]
                if hasattr(first_block, "conv1"):
                    original_filters = first_block.conv1.weight.size(0)
                    pruned_filters = self.pruned_neurons_count.get(layer_name, 0)
                    new_filters = original_filters - pruned_filters
                    pruned_indices = []
                    layer_seq = getattr(old_net, layer_name)
                    for block in layer_seq:
                        if hasattr(block, "conv1") and hasattr(
                            block.conv1, "weight_mask"
                        ):
                            zero_filters = torch.where(
                                block.conv1.weight_mask.sum(dim=(1, 2, 3)) == 0
                            )[0]
                            pruned_indices.extend(zero_filters.tolist())
                    pruned_indices = sorted(set(pruned_indices))
                    keep_indices = [
                        i for i in range(original_filters) if i not in pruned_indices
                    ]
                    pruning_config[layer_name] = {
                        "original_filters": original_filters,
                        "new_filters": new_filters,
                        "pruned_indices": pruned_indices,
                        "keep_indices": keep_indices,
                    }
                    print(
                        f"  {layer_name}: {len(pruned_indices)} neurons pruned, {len(keep_indices)} remaining"
                    )
        if not pruning_config:
            print("No layers with sufficient pruning detected.")
            return False
        print("\nOriginal network structure:")
        for layer_name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            if layer_name == "conv1" and hasattr(old_net, layer_name):
                print(
                    f"  {layer_name}: in={old_net.conv1.weight.size(1)}, out={old_net.conv1.weight.size(0)}"
                )
            elif hasattr(old_net, layer_name) and not isinstance(
                getattr(old_net, layer_name), nn.Identity
            ):
                layer = getattr(old_net, layer_name)[0]
                if hasattr(layer, "conv1"):
                    print(
                        f"  {layer_name}: in={layer.conv1.weight.size(1)}, out={layer.conv1.weight.size(0)}"
                    )
        new_net = self._build_custom_resnet(pruning_config)
        new_net = new_net.to(self.device)
        self._transfer_weights(old_net, new_net, pruning_config)
        new_net = self.update_classifier_after_pruning(new_net)
        self.net = new_net
        if hasattr(old_net, "dead_neuron_history"):
            self.net.dead_neuron_history = old_net.dead_neuron_history
        if hasattr(self, "opt"):
            if isinstance(self.opt, torch.optim.Adam):
                self.opt = torch.optim.Adam(
                    self.net.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.optim_wd,
                )
            else:  # SGD is the default
                self.opt = torch.optim.SGD(
                    self.net.parameters(),
                    lr=self.args.lr,
                    weight_decay=self.args.optim_wd,
                    momentum=self.args.optim_mom,
                )
        new_params = sum(p.numel() for p in new_net.parameters())
        reduction = (1 - new_params / original_params) * 100
        print(f"Original parameters: {original_params:,}")
        print(f"New parameters: {new_params:,}")
        print(f"Parameter reduction: {reduction:.2f}%")
        self.pruned_neurons_count = {}
        self.update_dead_neuron_tracking_after_pruning()
        return True

    def end_epoch(self, epoch, dataset):
        print(f"Epoch {epoch}:")
        # self.net.check_inactive_neurons(show_histogram=True)
        for layer_name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            # for layer_name in ["layer3", "layer4"]:
            self.net.analyze_layer_activations(
                layer_name, self.active_task, epoch, dead_threshold=0.05, top_k=10
            )

            if epoch != 0 and epoch % 10 == 0:
                self.reinitialize_reviving_neurons_method2(layer_name)
                self.prune_dead_neurons(layer_name)

        total_pruned = (
            sum(self.pruned_neurons_count.values())
            if hasattr(self, "pruned_neurons_count")
            else 0
        )
        if total_pruned > 0:
            self.verify_pruning_effectiveness()
            self.rebuild_network_with_pruning()
        self.fix_dimension_mismatches(debug=False)

    def end_task(self, dataset):
        """
        At the end of each task, prune neurons gradually and then rebuild the network.
        """
        # self.net.check_inactive_neurons(threshold=0.2)
        print(f"\n=== End of Task {self.active_task} Analysis ===")
        # if self.active_task > 0:
        # Prune neurons gradually for each layer
        for layer_name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            if layer_name in self.net.dead_neuron_history:
                self.net.analyze_dead_neuron_consistency(layer_name)
                # self.reinitialize_reviving_neurons(layer_name)

                # if self.active_task % 2:
                # self.prune_dead_neurons(layer_name)
        # self.net.save_dead_neuron_data("layer4")

        # total_pruned = (
        #     sum(self.pruned_neurons_count.values())
        #     if hasattr(self, "pruned_neurons_count")
        #     else 0
        # )
        # if total_pruned > 0:
        #     self.verify_pruning_effectiveness()
        #     self.rebuild_network_with_pruning()
        # self.fix_dimension_mismatches(debug=False)
        self.active_task += 1

    def begin_task(self, dataset):
        if self.active_task > 0:
            # for layer_name in ["conv1", "layer1", "layer2", "layer3", "layer4"]:
            # if layer_name in self.net.dead_neuron_history:
            #     self.reinitialize_reviving_neurons_method2(layer_name)
            # self.net.find_persistently_dead_neurons_accross_tasks_accross_tasks(
            #     active_task=self.active_task, layer_name=layer_name
            # )
            # self.net.find_persistently_reviving_neurons(
            #     active_task=self.active_task, layer_name=layer_name
            # )
            # self.prune_dead_neurons(layer_name)

            self.net.save_combined_neuron_analysis(self.active_task)
        # total_pruned = (
        #     sum(self.pruned_neurons_count.values())
        #     if hasattr(self, "pruned_neurons_count")
        #     else 0
        # )
        # if total_pruned > 0:
        #     self.verify_pruning_effectiveness()
        #     self.rebuild_network_with_pruning()
        # self.fix_dimension_mismatches(debug=False)

    def observe_sam(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains with gradient clipping and loss checking to prevent NaN values.
        """
        if hasattr(self, "pruned_neurons_count") and self.pruned_neurons_count:
            self.fix_dimension_mismatches(debug=False)

        enable_running_stats(self.net)
        outputs = self.net(inputs)
        loss = smooth_crossentropy(outputs, labels).mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        disable_running_stats(self.net)
        smooth_crossentropy(self.net(inputs), labels).mean().backward()
        self.optimizer.second_step(zero_grad=True)
        return loss.item()

    def observe_normal(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains with gradient clipping and loss checking to prevent NaN values.
        """
        if hasattr(self, "pruned_neurons_count") and self.pruned_neurons_count:
            self.fix_dimension_mismatches(debug=False)

        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        self.opt.step()

        return loss.item()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, **kwargs):
        """
        SGD trains with gradient clipping and loss checking to prevent NaN values.
        """
        return self.observe_normal(inputs, labels, not_aug_inputs, epoch, **kwargs)
