"""
This module implements the simplest form of rehearsal training: Experience Replay. It maintains a buffer
of previously seen examples and uses them to augment the current batch during training.

Example usage:
    model = Er(backbone, loss, args, transform, dataset)
    loss = model.observe(inputs, labels, not_aug_inputs, epoch)

"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class Er_Thesis(ContinualModel):
    """Continual learning via Experience Replay."""

    NAME = "er-thesis"
    COMPATIBILITY = ["class-il", "domain-il", "task-il", "general-continual"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser object with predefined arguments for the Er model.

        This model requires the `add_rehearsal_args` to include the buffer-related arguments.
        """
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        The ER model maintains a buffer of previously seen examples and uses them to augment the current batch during training.
        """
        super(Er_Thesis, self).__init__(
            backbone, loss, args, transform, dataset=dataset
        )
        self.buffer = Buffer(self.args.buffer_size)
        self.iteration = 0
        print("Er model initialized")

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        ER trains on the current task using the data provided, but also augments the batch with data from the buffer.
        """
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss += 1e-8
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        # Apply pruning at the end of each epoch
        if epoch is not None and self.iteration != epoch:
            print(f"Epoch {epoch}:")

            # Check inactive neurons
            if epoch % 1 == 0:
                # self.apply_neuron_death(epoch)

                inactive_stats = self.net.check_inactive_neurons()
                self.net.analyze_layer_activations("layer3", top_k=10)
                self.net.analyze_layer_activations("layer4", top_k=10)

        self.iteration = epoch

        return loss.item()

    def apply_neuron_death(self, epoch):
        """
        Applies artificial neuron death to layer4 and layer3 by masking out weights and biases.

        Args:
            epoch (int): Current training epoch
        """
        with torch.no_grad():
            layer_masks = {}

            # First pass: Create masks for both layers
            for name, param in self.net.named_parameters():
                if "layer3" in name and "weight" in name:
                    out_channels = param.shape[0]
                    # Create persistent mask for layer3
                    if "layer3" not in layer_masks:
                        layer_masks["layer3"] = (
                            torch.rand(out_channels, device=param.device) > 0.7
                        )

                    # Apply mask to all weights for this channel
                    expanded_mask = layer_masks["layer3"].view(
                        -1, *([1] * (len(param.shape) - 1))
                    )
                    param.data *= expanded_mask
                    print(f"Layer3 weight shape: {param.shape}")
                    print(
                        f"Layer3 channels killed: {(~layer_masks['layer3']).sum().item()}/{out_channels}"
                    )

                if "layer4" in name and "weight" in name:
                    out_channels = param.shape[0]
                    # Create persistent mask for layer4
                    if "layer4" not in layer_masks:
                        layer_masks["layer4"] = (
                            torch.rand(out_channels, device=param.device) > 0.5
                        )

                    # Apply mask to all weights for this channel
                    expanded_mask = layer_masks["layer4"].view(
                        -1, *([1] * (len(param.shape) - 1))
                    )
                    param.data *= expanded_mask
                    print(f"layer4 weight shape: {param.shape}")
                    print(
                        f"layer4 channels killed: {(~layer_masks['layer4']).sum().item()}/{out_channels}"
                    )

            # Second pass: Zero out biases and ensure complete neuron death
            for name, param in self.net.named_parameters():
                if "bias" in name:
                    if "layer3" in name and "layer3" in layer_masks:
                        param.data *= layer_masks["layer3"]
                    elif "layer4" in name and "layer4" in layer_masks:
                        param.data *= layer_masks["layer4"]

            # Force complete neuron death by setting all parameters to exactly zero
            for name, param in self.net.named_parameters():
                if "layer3" in name:
                    if "weight" in name:
                        dead_channels = ~layer_masks["layer3"]
                        param.data[dead_channels] = 0.0
                if "layer4" in name:
                    if "weight" in name:
                        dead_channels = ~layer_masks["layer4"]
                        param.data[dead_channels] = 0.0

            print(f"Applied artificial neuron death at epoch {epoch}")
