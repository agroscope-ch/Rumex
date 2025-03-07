import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

import torch
from torchvision.ops import box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def evaluate(model, data_loader, device, iou_thresholds=(0.5, 0.75), conf_threshold=0.5):
    """
    Enhanced evaluation function for object detection models
    
    Args:
        model: The PyTorch detection model to evaluate
        data_loader: DataLoader containing validation/test data
        device: Device to run evaluation on
        iou_thresholds: IoU thresholds for evaluation (can be single value or list)
        conf_threshold: Confidence threshold for filtering detections
        
    Returns:
        Dict containing comprehensive evaluation metrics
    """
    model.eval()
    
    # If a single IoU threshold is provided, convert to list
    if isinstance(iou_thresholds, (int, float)):
        iou_thresholds = [iou_thresholds]
    
    # Initialize overall metrics
    map_metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_thresholds=list(iou_thresholds),
        max_detection_thresholds=[1, 10, 100]
    )
    
    # Track detailed metrics
    class_metrics = defaultdict(lambda: {
        'TP': 0, 'FP': 0, 'FN': 0, 
        'total_predictions': 0, 'total_targets': 0,
        'confidences': []
    })
    
    total_loss = 0.0
    num_images = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            num_images += len(images)
            
            # Get predictions
            predictions = model(images)
            
            # Calculate loss
            try:
                loss = model.loss(predictions, targets)
                total_loss += loss.item()
            except Exception as e:
                print(f"Warning: Could not compute loss. Error: {e}")
            
            # Update mAP metric (no need to filter by confidence here as the metric handles it)
            map_metric.update(predictions, targets)
            
            # Calculate per-class metrics for each image
            for pred, target in zip(predictions, targets):
                # Apply confidence threshold
                mask = pred['scores'] > conf_threshold
                filtered_boxes = pred['boxes'][mask]
                filtered_labels = pred['labels'][mask]
                filtered_scores = pred['scores'][mask]
                
                target_boxes = target['boxes']
                target_labels = target['labels']
                
                # Add to the confidence score distribution (for PR curve)
                for label, score in zip(filtered_labels.cpu().numpy(), filtered_scores.cpu().numpy()):
                    label_id = int(label)
                    class_metrics[label_id]['confidences'].append(float(score))
                
                # Perform detailed metrics calculation
                update_per_class_metrics(
                    filtered_boxes, filtered_labels, 
                    target_boxes, target_labels, 
                    iou_thresholds[0], class_metrics
                )
    
    # Compute mAP metrics
    map_results = map_metric.compute()
    
    # Process class-specific metrics
    processed_class_metrics = {}
    for class_id, metrics in class_metrics.items():
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        
        # Calculate precision, recall, F1 for this class
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        processed_class_metrics[f'class_{class_id}'] = {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': metrics['total_predictions'],
            'targets': metrics['total_targets']
        }
    
    # Calculate micro-average metrics (across all classes)
    total_TP = sum(metrics['TP'] for metrics in class_metrics.values())
    total_FP = sum(metrics['FP'] for metrics in class_metrics.values())
    total_FN = sum(metrics['FN'] for metrics in class_metrics.values())
    
    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Calculate macro-average metrics (average of per-class metrics)
    class_precisions = [m['precision'] for m in processed_class_metrics.values()]
    class_recalls = [m['recall'] for m in processed_class_metrics.values()]
    class_f1s = [m['f1'] for m in processed_class_metrics.values()]
    
    macro_precision = sum(class_precisions) / len(class_precisions) if class_precisions else 0
    macro_recall = sum(class_recalls) / len(class_recalls) if class_recalls else 0
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0
    
    # Compile final results
    results = {
        # mAP metrics
        'mAP': map_results['map'].item(),
        'mAP_50': map_results['map_50'].item(),
        'mAP_75': map_results['map_75'].item(),
        'mAP_small': map_results['map_small'].item(),
        'mAP_medium': map_results['map_medium'].item(),
        'mAP_large': map_results['map_large'].item(),
        
        # Overall metrics
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'total_TP': total_TP,
        'total_FP': total_FP,
        'total_FN': total_FN,
        
        # Loss
        'loss': total_loss / len(data_loader) if len(data_loader) > 0 else float('inf'),
        
        # Number of samples evaluated
        'num_images': num_images,
        
        # Per-class metrics
        'class_metrics': processed_class_metrics
    }
    
    return results


def update_per_class_metrics(pred_boxes, pred_labels, target_boxes, target_labels, iou_threshold, class_metrics):
    """
    Updates the per-class metrics by comparing predictions with targets
    
    Args:
        pred_boxes: Predicted bounding boxes [x1, y1, x2, y2]
        pred_labels: Predicted class labels
        target_boxes: Target bounding boxes
        target_labels: Target class labels
        iou_threshold: IoU threshold for matching
        class_metrics: Dictionary to update with metrics
    """
    if len(pred_boxes) == 0:
        # Handle case with no predictions
        for target_label in target_labels.unique():
            label_id = int(target_label.item())
            class_metrics[label_id]['FN'] += (target_labels == target_label).sum().item()
            class_metrics[label_id]['total_targets'] += (target_labels == target_label).sum().item()
        return
    
    if len(target_boxes) == 0:
        # Handle case with no targets
        for pred_label in pred_labels.unique():
            label_id = int(pred_label.item())
            class_metrics[label_id]['FP'] += (pred_labels == pred_label).sum().item()
            class_metrics[label_id]['total_predictions'] += (pred_labels == pred_label).sum().item()
        return
    
    # Compute IoU matrix between all pred and target boxes
    iou_matrix = box_iou(pred_boxes, target_boxes)
    
    # For each prediction, find the best matching target of the same class
    for pred_idx, pred_label in enumerate(pred_labels):
        pred_label_item = int(pred_label.item())
        class_metrics[pred_label_item]['total_predictions'] += 1
        
        # Find targets with the same class
        matching_targets = (target_labels == pred_label)
        
        if not matching_targets.any():
            # No targets of this class, count as FP
            class_metrics[pred_label_item]['FP'] += 1
            continue
            
        # Get IoUs with targets of the same class
        valid_ious = iou_matrix[pred_idx, matching_targets]
        
        # If no valid IoUs above threshold, count as FP
        if valid_ious.shape[0] == 0 or valid_ious.max() < iou_threshold:
            class_metrics[pred_label_item]['FP'] += 1
            continue
            
        # Found a match - this is a true positive
        class_metrics[pred_label_item]['TP'] += 1
        
        # Mark this target as matched (for FN calculation)
        target_idx = torch.where(matching_targets)[0][valid_ious.argmax()]
        matching_targets[target_idx] = False
    
    # Count unmatched targets as false negatives
    for target_label in target_labels.unique():
        target_label_item = int(target_label.item())
        class_count = (target_labels == target_label).sum().item()
        class_metrics[target_label_item]['total_targets'] += class_count
        
        # Calculate FN as total_targets - TP for this class in this image
        matched = min(class_metrics[target_label_item]['TP'], class_count)
        class_metrics[target_label_item]['FN'] += class_count - matched


def visualize_pr_curves(results, class_names=None):
    """
    Visualize precision-recall curves from evaluation results
    
    Args:
        results: Results dictionary from evaluate()
        class_names: Optional mapping from class IDs to names
    """
    import matplotlib.pyplot as plt
    
    # Create subplots - one for micro-average and one for each class
    num_classes = len(results['class_metrics'])
    fig, axs = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))
    
    # Plot micro-average PR curve
    axs[0].plot([results['micro_recall']], [results['micro_precision']], 'ro', markersize=8)
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].set_title(f'Micro-Average\nF1={results["micro_f1"]:.3f}')
    axs[0].grid(True)
    
    # Plot per-class PR curves
    for i, (class_id, metrics) in enumerate(results['class_metrics'].items()):
        class_name = class_names[int(class_id.split('_')[1])] if class_names else class_id
        
        axs[i+1].plot([metrics['recall']], [metrics['precision']], 'bo', markersize=8)
        axs[i+1].set_xlim([0, 1])
        axs[i+1].set_ylim([0, 1])
        axs[i+1].set_xlabel('Recall')
        axs[i+1].set_ylabel('Precision')
        axs[i+1].set_title(f'{class_name}\nF1={metrics["f1"]:.3f}')
        axs[i+1].grid(True)
    
    plt.tight_layout()
    return fig


def print_metrics_report(results, class_names=None):
    """
    Print a comprehensive metrics report
    
    Args:
        results: Results dictionary from evaluate()
        class_names: Optional mapping from class IDs to names
    """
    print("=" * 50)
    print("DETECTION METRICS REPORT")
    print("=" * 50)
    
    print("\nMEAN AVERAGE PRECISION:")
    print(f"mAP: {results['mAP']:.4f}")
    print(f"mAP@0.5: {results['mAP_50']:.4f}")
    print(f"mAP@0.75: {results['mAP_75']:.4f}")
    print(f"mAP (small): {results['mAP_small']:.4f}")
    print(f"mAP (medium): {results['mAP_medium']:.4f}")
    print(f"mAP (large): {results['mAP_large']:.4f}")
    
    print("\nOVERALL METRICS:")
    print(f"Micro Precision: {results['micro_precision']:.4f}")
    print(f"Micro Recall: {results['micro_recall']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    
    print("\nCONFUSION MATRIX ELEMENTS:")
    print(f"Total TP: {results['total_TP']}")
    print(f"Total FP: {results['total_FP']}")
    print(f"Total FN: {results['total_FN']}")
    
    print("\nPER-CLASS METRICS:")
    for class_id, metrics in results['class_metrics'].items():
        class_name = class_names[int(class_id.split('_')[1])] if class_names else class_id
        print(f"\n{class_name}:")
        print(f"  TP: {metrics['TP']}")
        print(f"  FP: {metrics['FP']}")
        print(f"  FN: {metrics['FN']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Total Predictions: {metrics['predictions']}")
        print(f"  Total Targets: {metrics['targets']}")
    
    print("\nEVALUATION INFO:")
    print(f"Loss: {results['loss']:.4f}")
    print(f"Images Evaluated: {results['num_images']}")
    print("=" * 50)


def evaluate_map50(model, val_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate the model and compute the mAP@50.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): The data loader for the validation dataset.
        device (torch.device): The device to move the model and data to.
        iou_threshold (float): The IoU threshold for considering a detection as a true positive.
        conf_threshold (float): The confidence threshold for filtering predictions.

    Returns:
        float: The mAP@50 value.
    """
    model.eval()

    # Initialize the Mean Average Precision metric
    map_metric = MeanAveragePrecision(iou_thresholds=[iou_threshold])

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Filter predictions based on confidence threshold
            preds = []
            for pred in predictions:
                mask = pred['scores'] > conf_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'scores': pred['scores'][mask],
                    'labels': pred['labels'][mask]
                }
                preds.append(filtered_pred)

            # Update the metric with predictions and targets
            map_metric.update(preds, targets)

    # Compute the mAP@50
    map50 = map_metric.compute()['map_50']
    return map50



def evaluate(model, val_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluation step for the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (DataLoader): The data loader for the validation dataset.
        device (torch.device): The device to move the model and data to.
        iou_threshold (float): The IoU threshold for considering a detection as a true positive.
        conf_threshold (float): The confidence threshold for filtering predictions.

    Returns:
        dict: A dictionary containing the average metrics.
    """
    model.eval()
    total_metrics = {}

    # Initialize metrics
    map_metric = MeanAveragePrecision(box_format='xyxy')

    total_loss = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    total_precision = 0
    total_recall = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Calculate loss
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()

            # Calculate metrics for each image
            for pred, target in zip(predictions, targets):
                metrics, fp, fn, tp, precision, recall = calculate_metrics(
                    pred['boxes'],
                    pred['labels'],
                    pred['scores'],
                    target['boxes'],
                    target['labels'],
                    iou_threshold,
                    conf_threshold,
                    map_metric
                )

                # Accumulate metrics
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = []
                    total_metrics[k].append(v.item())

                total_fp += fp
                total_fn += fn
                total_tp += tp
                total_precision += precision
                total_recall += recall

    avg_metrics = {k: sum(v) / len(v) for k, v in total_metrics.items()}
    avg_metrics['FP'] = total_fp / len(val_loader)
    avg_metrics['FN'] = total_fn / len(val_loader)
    avg_metrics['TP'] = total_tp / len(val_loader)
    avg_metrics['Precision'] = total_precision / len(val_loader)
    avg_metrics['Recall'] = total_recall / len(val_loader)
    avg_metrics['F1-score'] = 2 * (avg_metrics['Precision'] * avg_metrics['Recall']) / (avg_metrics['Precision'] + avg_metrics['Recall'])
    avg_metrics['Eval Loss'] = total_loss / len(val_loader)

    return avg_metrics

def calculate_metrics(pred_boxes, pred_labels, pred_scores, true_boxes, true_labels, iou_threshold, conf_threshold, map_metric):
    """
    Calculate detection metrics.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes.
        pred_labels (torch.Tensor): Predicted labels.
        pred_scores (torch.Tensor): Predicted scores.
        true_boxes (torch.Tensor): True bounding boxes.
        true_labels (torch.Tensor): True labels.
        iou_threshold (float): The IoU threshold for considering a detection as a true positive.
        conf_threshold (float): The confidence threshold for filtering predictions.
        map_metric (MeanAveragePrecision): The Mean Average Precision metric.

    Returns:
        tuple: A tuple containing the metrics, false positives, false negatives, true positives, precision, and recall.
    """
    # Filter predictions based on confidence threshold
    mask = pred_scores > conf_threshold
    pred_boxes = pred_boxes[mask]
    pred_labels = pred_labels[mask]
    pred_scores = pred_scores[mask]

    preds = [dict(
        boxes=pred_boxes,
        scores=pred_scores,
        labels=pred_labels,
    )]

    target = [dict(
        boxes=true_boxes,
        labels=true_labels,
    )]

    map_metric.update(preds, target)
    metrics = map_metric.compute()

    # Calculate TP, FP, FN
    ious = box_iou(pred_boxes, true_boxes)
    max_ious, matched_indices = ious.max(dim=1)

    tp = (max_ious > iou_threshold).sum().item()
    fp = (max_ious <= iou_threshold).sum().item()
    fn = len(true_boxes) - tp

    # Calculate Precision and Recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    metrics.update({
        'Precision': precision,
        'Recall': recall,
    })

    return metrics, fp, fn, tp, precision, recall
