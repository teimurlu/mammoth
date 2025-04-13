import os
import sys
import torch
import pytest
from torch.nn import CrossEntropyLoss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main
from models.er_thesis import Er_Thesis
from backbone.ResNetBlock import resnet18


class TestErThesis:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Setup for each test"""

        # Mock arguments
        class Args:
            def __init__(self):
                self.buffer_size = 200
                self.minibatch_size = 32
                self.debug_mode = 1
                self.num_workers = 0
                self.num_classes = 10
                self.nf = 64
                self.dataset = "cifar10"
                self.transform_type = "default"
                self.dataset_path = "data"

        self.args = Args()
        self.backbone = resnet18(num_classes=self.args.num_classes)
        self.model = Er_Thesis(
            backbone=self.backbone,
            loss=CrossEntropyLoss(),
            args=self.args,
            transform=None,
        )
        self.test_input = torch.randn(32, 3, 32, 32)

        self.model.eval()

    def test_inactive_detection_basic(self):
        """Test basic inactive neuron detection"""
        percentage = self.model.net.check_inactive_neurons(self.test_input)
        assert isinstance(percentage, float)
        assert 0 <= percentage <= 100

    def test_inactive_history_tracking(self):
        """Test tracking of consistently inactive neurons"""
        # First pass
        self.model.net.check_inactive_neurons(self.test_input)
        self.model.net.update_inactive_history()
        first_ratios = self.model.net.get_consistently_inactive_ratio()

        # Second pass with modified input
        modified_input = self.test_input * 0.5  # Scale down activations
        self.model.net.check_inactive_neurons(modified_input)
        self.model.net.update_inactive_history()
        second_ratios = self.model.net.get_consistently_inactive_ratio()

        assert len(first_ratios) == len(second_ratios)

    def test_artificial_zeroing(self):
        """Test detection with artificially zeroed neurons"""
        # Get initial metrics
        initial_percentage = self.model.net.check_inactive_neurons(self.test_input)

        # Artificially zero out activations
        for name, activation in self.model.net.activations.items():
            mask = torch.rand_like(activation) > 0.5
            self.model.net.activations[name][mask] = 0

        # Check modified metrics
        modified_percentage = self.model.net.check_inactive_neurons(self.test_input)
        assert modified_percentage > initial_percentage

    @pytest.mark.parametrize(
        "dataset", ["seq-mnist", "seq-cifar10", "rot-mnist", "perm-mnist", "mnist-360"]
    )
    def test_er(self, dataset):
        """Original ER test"""
        sys.argv = [
            "mammoth",
            "--model",
            "er",
            "--dataset",
            dataset,
            "--buffer_size",
            "10",
            "--lr",
            "1e-4",
            "--n_epochs",
            "1",
            "--batch_size",
            "2",
            "--non_verbose",
            "1",
            "--num_workers",
            "0",
            "--seed",
            "0",
            "--debug_mode",
            "1",
        ]

        main()
