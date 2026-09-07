"""
Unit Test Suite for Monocular Depth Estimation & Quadrant Risk Analysis
Author: Bhanu Vignesh Naidu Ganeshna
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.models import DepthCNN, build_model
from src.losses import (
    scale_invariant_log_loss,
    compute_abs_rel,
    compute_rmse,
    MultiTaskQuadrantLoss
)
from src.dataset import (
    DepthDataset,
    get_transforms,
    LABEL_MAP,
    IDX_MAP,
    QUADRANT_NAMES
)
from src.visualizer import create_quadrant_overlay
from src.evaluator import evaluate_model, format_evaluation_report
from main import build_parser


class TestScaleInvariantLoss(unittest.TestCase):
    """Unit tests for scale-invariant logarithmic depth loss."""

    def test_identical_depths_zero_loss(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        loss = scale_invariant_log_loss(depth, depth)
        self.assertAlmostEqual(loss, 0.0, places=5)

    def test_scale_invariance_property(self):
        # A uniform scale factor c should yield identical loss
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.2, 1.8], [3.3, 3.8]])
        loss1 = scale_invariant_log_loss(y_true, y_pred)

        # Scale predicted depth by constant multiplier c=2.5
        loss2 = scale_invariant_log_loss(y_true, y_pred * 2.5)
        self.assertAlmostEqual(loss1, loss2, places=4)

    def test_torch_tensor_input(self):
        y_true = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        y_pred = torch.tensor([[1.8, 4.2], [5.9, 8.1]])
        loss = scale_invariant_log_loss(y_true, y_pred)
        self.assertGreater(loss, 0.0)

    def test_zero_depth_handling(self):
        y_true = np.zeros((4, 4))
        y_pred = np.zeros((4, 4))
        loss = scale_invariant_log_loss(y_true, y_pred)
        self.assertEqual(loss, 0.0)


class TestErrorMetrics(unittest.TestCase):
    """Unit tests for Abs Rel and RMSE depth metrics."""

    def test_compute_abs_rel_identical(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(compute_abs_rel(y, y), 0.0, places=5)

    def test_compute_abs_rel_known(self):
        y_true = np.array([2.0, 4.0])
        y_pred = np.array([1.0, 5.0])
        # |2-1|/2 = 0.5, |4-5|/4 = 0.25 -> mean = 0.375
        self.assertAlmostEqual(compute_abs_rel(y_true, y_pred), 0.375, places=5)

    def test_compute_rmse_identical(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(compute_rmse(y, y), 0.0, places=5)

    def test_compute_rmse_known(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([4.0, 6.0])
        # diffs: 3, 4 -> diff^2: 9, 16 -> mean: 12.5 -> sqrt(12.5) ~ 3.5355
        self.assertAlmostEqual(compute_rmse(y_true, y_pred), np.sqrt(12.5), places=4)


class TestDepthCNNArchitecture(unittest.TestCase):
    """Unit tests for Multi-Head ResNet-18 DepthCNN model."""

    def setUp(self):
        self.model = DepthCNN(num_classes=3, dropout_p=0.2, pretrained=False)
        self.model.eval()

    def test_forward_pass_shapes(self):
        x = torch.randn(2, 3, 224, 224)
        outputs = self.model(x)

        # Output must be a list of 4 tensors (TL, TR, BL, BR)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out.shape, (2, 3))

    def test_build_model_factory(self):
        m = build_model(num_classes=3, device="cpu")
        self.assertIsInstance(m, DepthCNN)


class TestSpatialDataset(unittest.TestCase):
    """Unit tests for dataset loading, transformations, and coordinate swaps."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.df = pd.DataFrame({
            "filename": ["img1.jpg", "img2.jpg"],
            "TL": ["N", "F"],
            "TR": ["M", "M"],
            "BL": ["F", "N"],
            "BR": ["M", "N"]
        })
        self.csv_path = os.path.join(self.temp_dir.name, "data.csv")
        self.df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dataset_length_and_types(self):
        dataset = DepthDataset(self.csv_path, img_dir=self.temp_dir.name, is_train=False)
        self.assertEqual(len(dataset), 2)

        img, labels, fname = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 224, 224))
        self.assertEqual(labels.shape, (4,))
        self.assertEqual(fname, "img1.jpg")

        # Check label mapping: 'N'=0, 'M'=1, 'F'=2
        # img1: TL='N'(0), TR='M'(1), BL='F'(2), BR='M'(1)
        expected_labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        self.assertTrue(torch.equal(labels, expected_labels))

    def test_label_mapping_constants(self):
        self.assertEqual(LABEL_MAP['N'], 0)
        self.assertEqual(LABEL_MAP['M'], 1)
        self.assertEqual(LABEL_MAP['F'], 2)
        self.assertEqual(len(QUADRANT_NAMES), 4)


class TestMultiTaskLoss(unittest.TestCase):
    """Unit tests for multi-task quadrant cross entropy loss."""

    def setUp(self):
        self.loss_fn = MultiTaskQuadrantLoss()

    def test_loss_computation(self):
        # 4 logits tensors each of shape (2, 3)
        logits = [torch.randn(2, 3, requires_grad=True) for _ in range(4)]
        targets = torch.tensor([[0, 1, 2, 0], [2, 2, 1, 0]], dtype=torch.long)

        loss = self.loss_fn(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0.0)

        # Verify backpropagation
        loss.backward()
        for head_logits in logits:
            self.assertIsNotNone(head_logits.grad)


class TestVisualizerOverlay(unittest.TestCase):
    """Unit tests for 4-quadrant spatial hazard overlay generator."""

    def test_create_quadrant_overlay_dimensions(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        preds = [0, 1, 2, 0]  # TL=Near, TR=Middle, BL=Far, BR=Near
        gts = ["Near", "Middle", "Far", "Near"]

        overlay = create_quadrant_overlay(canvas, preds, gts=gts, alpha=0.4)
        self.assertEqual(overlay.shape, (480, 640, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_create_quadrant_overlay_pil_image(self):
        pil_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        preds = [2, 2, 1, 0]
        overlay = create_quadrant_overlay(pil_img, preds)
        self.assertEqual(overlay.shape, (200, 200, 3))


class TestEvaluator(unittest.TestCase):
    """Unit tests for model evaluation and reporting."""

    def test_evaluate_model_dummy(self):
        model = DepthCNN(num_classes=3, pretrained=False)
        dummy_data = [(torch.randn(2, 3, 224, 224), torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]), ["f1", "f2"])]
        results = evaluate_model(model, dummy_data, torch.device("cpu"))

        self.assertIn("overall_accuracy", results)
        self.assertIn("quadrant_accuracies", results)
        self.assertEqual(len(results["quadrant_accuracies"]), 4)

        report = format_evaluation_report(results)
        self.assertIn("4-QUADRANT SPATIAL RISK EVALUATION REPORT", report)


class TestCLIArgumentParsing(unittest.TestCase):
    """Unit tests for CLI parser."""

    def setUp(self):
        self.parser = build_parser()

    def test_default_cli_arguments(self):
        args = self.parser.parse_args([])
        self.assertEqual(args.mode, "metrics")
        self.assertEqual(args.data, "dataset.csv")
        self.assertEqual(args.test_csv, "test.csv")
        self.assertEqual(args.model, "best_model.pth")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch, 8)

    def test_custom_cli_arguments(self):
        cmd = ["--mode", "eval", "--test-csv", "custom_test.csv", "--batch", "16", "--device", "cpu"]
        args = self.parser.parse_args(cmd)
        self.assertEqual(args.mode, "eval")
        self.assertEqual(args.test_csv, "custom_test.csv")
        self.assertEqual(args.batch, 16)
        self.assertEqual(args.device, "cpu")


class TestTrainer(unittest.TestCase):
    """Unit tests for training routine."""

    def test_train_epoch_dummy(self):
        from src.trainer import train_epoch
        from src.losses import MultiTaskQuadrantLoss
        model = DepthCNN(num_classes=3, pretrained=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = MultiTaskQuadrantLoss()
        dummy_loader = [(torch.randn(2, 3, 224, 224), torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]]), ["f1", "f2"])]

        loss = train_epoch(model, dummy_loader, optimizer, criterion, torch.device("cpu"))
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
