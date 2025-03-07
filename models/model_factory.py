import torch
import torchvision
from torchvision.models.detection import FasterRCNN, RetinaNet, FCOS
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.fcos import FCOSHead

def init_model(model_name, backbone_name, num_classes, device, weights=None, train_backbone=False):
    """
    Initialize the object detection model and move it to the specified device.

    Args:
        model_name (str): Name of the model (e.g., 'fasterrcnn', 'retinanet', 'ssd', 'fcos').
        backbone_name (str): Name of the backbone (e.g., 'resnet50', 'mobilenet_v3_large').
        num_classes (int): Number of classes (including background).
        device (torch.device): The device to move the model to.
        weights (str or None): Pre-trained weights.
        train_backbone (bool): Whether to train the backbone.

    Returns:
        model (torch.nn.Module): The initialized model.
    """
    # Map model names to their respective functions
    model_map = {
        'fasterrcnn': {
            'resnet50': torchvision.models.detection.fasterrcnn_resnet50_fpn,
            'mobilenet_v3_large_320': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            'mobilenet_v3_large': torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
        },
        'fasterrcnnV2': torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
        'retinanet': torchvision.models.detection.retinanet_resnet50_fpn,
    }

    # Map backbone names to their respective weights
    backbone_map = {
        'resnet50': {
            'fasterrcnn': torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights,
            'fasterrcnnV2': torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights,
            'retinanet': torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights,
        },
        'mobilenet_v3_large_320': {
            'fasterrcnn': torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        },
        'mobilenet_v3_large': {
            'fasterrcnn': torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights
        },
    }

    # Validate model and backbone names
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} is not supported.")
    if backbone_name not in backbone_map:
        raise ValueError(f"Backbone {backbone_name} is not supported.")
    if isinstance(model_map[model_name], dict) and backbone_name not in model_map[model_name]:
        raise ValueError(f"Backbone {backbone_name} is not supported for model {model_name}.")

    # Initialize the model with the specified backbone and weights
    if isinstance(model_map[model_name], dict):
        model_func = model_map[model_name][backbone_name]
    else:
        model_func = model_map[model_name]

    weights_class = backbone_map[backbone_name][model_name]
    weights_instance = weights_class.DEFAULT if weights is None else getattr(weights_class, weights)

    model = model_func(weights='COCO_V1')

    # Modify the model's head based on the model type
    if model_name in ['fasterrcnn', 'fasterrcnnV2']:
        # Replace the FastRCNN head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    elif model_name == 'retinanet':
        # Replace the RetinaNet head
        model.head = RetinaNetHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
    elif model_name == 'ssd' or model_name == 'ssdlite':
        # Replace the SSD head
        model.head.classification_head = SSDClassificationHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
        )
    elif model_name == 'fcos':
        # Replace the FCOS head
        model.head.classification_head = FCOSHead(
            in_channels=model.backbone.out_channels,
            num_classes=num_classes,
        )

    # Optionally freeze the backbone
    if not train_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Move the model to the specified device
    model = model.to(device)
    return model

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define number of classes (update this for your dataset)
    num_classes = 2  # e.g., background + bird

    # Initialize model with transfer learning settings
    model = init_model(
        model_name='fasterrcnn',
        backbone_name='resnet50',
        num_classes=num_classes,
        device=device,
        weights="COCO_V1",
        train_backbone=False
    )

    # Print model summary
    print("\nModel Overview:")
    print(f"Model type: Faster R-CNN with ResNet50 backbone")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
