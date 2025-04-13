# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class EwcOn(ContinualModel):
    """Continual learning via online EWC."""

    NAME = "ewc_on"
    COMPATIBILITY = ["class-il", "domain-il", "task-il"]

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument(
            "--e_lambda", type=float, required=True, help="lambda weight for EWC"
        )
        parser.add_argument(
            "--gamma", type=float, required=True, help="gamma parameter for EWC online"
        )
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(EwcOn, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.fish = None

        self.iteration = 0
        self.active_task = 0

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (
                self.args.e_lambda
                * (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            )
            return penalty

    def end_task(self, dataset):

        self.net.check_inactive_neurons()
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            self.net.analyze_layer_activations(
                layer_name, self.active_task, self.iteration, top_k=10
            )

        print(f"\n=== End of Task {self.active_task} Analysis ===")
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            if layer_name in self.net.dead_neuron_history:
                self.net.analyze_dead_neuron_consistency(layer_name)

        self.active_task += 1
        self.iteration = 0

        self.net.save_dead_neuron_data("layer4")

        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            print(f"Computing Fisher: {j}/{len(dataset.train_loader)} \r", end="")
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = -F.nll_loss(
                    self.logsoft(output), lab.unsqueeze(0), reduction="none"
                )
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= len(dataset.train_loader) * self.args.batch_size

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        print("Fisher computed")
        self.checkpoint = self.net.get_params().data.clone()

    def get_penalty_grads(self):
        return (
            self.args.e_lambda
            * 2
            * self.fish
            * (self.net.get_params().data - self.checkpoint)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        if self.checkpoint is not None:
            self.net.set_grads(self.get_penalty_grads())
        loss = self.loss(outputs, labels)
        # loss += 1e-8
        # assert not torch.isnan(loss) # WAS ORIGINALLY HERE

        if torch.isnan(loss):
            print("Warning: NaN loss detected")
            return 0.0

        # Add penalty term more safely
        if self.checkpoint is not None:
            penalty_grads = self.get_penalty_grads()
            # Clip penalty gradients
            penalty_grads = torch.clamp(penalty_grads, -1.0, 1.0)
            self.net.set_grads(penalty_grads)

        loss.backward()

        # Clip gradients after backward pass
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

        # self.opt.step()
        # loss.backward()
        self.opt.step()

        if epoch is not None and (self.iteration != epoch):
            print(f"Epoch {epoch}:")

            # if epoch % 1 == 0:
            self.net.check_inactive_neurons()
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                self.net.analyze_layer_activations(
                    layer_name, self.active_task, self.iteration, top_k=10
                )

        self.iteration = epoch

        return loss.item()
