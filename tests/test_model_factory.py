import unittest
import torch
from models.model_factory import init_model

class TestModelFactory(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2  # Example number of classes

    def test_fasterrcnn_resnet50(self):
        model = init_model(
            model_name='fasterrcnn',
            backbone_name='resnet50',
            num_classes=self.num_classes,
            device=self.device,
            weights="COCO_V1",
            train_backbone=False
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(all(param.requires_grad is False for param in model.backbone.parameters()))


    def test_fasterrcnn_mobilenet_v3_large_320(self):
        model = init_model(
            model_name='fasterrcnn',
            backbone_name='mobilenet_v3_large_320',
            num_classes=self.num_classes,
            device=self.device,
            weights="COCO_V1",
            train_backbone=False
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(all(param.requires_grad is False for param in model.backbone.parameters()))

    def test_fasterrcnn_mobilenet_v3_large(self):
        model = init_model(
            model_name='fasterrcnn',
            backbone_name='mobilenet_v3_large',
            num_classes=self.num_classes,
            device=self.device,
            weights="COCO_V1",
            train_backbone=False
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(all(param.requires_grad is False for param in model.backbone.parameters()))

    def test_fasterrcnnV2_resnet50(self):
        model = init_model(
            model_name='fasterrcnnV2',
            backbone_name='resnet50',
            num_classes=self.num_classes,
            device=self.device,
            weights="COCO_V1",
            train_backbone=False
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(all(param.requires_grad is False for param in model.backbone.parameters()))

    def test_retinanet_resnet50(self):
        model = init_model(
            model_name='retinanet',
            backbone_name='resnet50',
            num_classes=self.num_classes,
            device=self.device,
            weights="COCO_V1",
            train_backbone=False
        )
        self.assertIsInstance(model, torch.nn.Module)
        self.assertTrue(all(param.requires_grad is False for param in model.backbone.parameters()))

if __name__ == '__main__':
    unittest.main()
