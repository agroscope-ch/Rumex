import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_best_model(model, best_model_path):
    """
    Load the best model from checkpoint.
    """
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with mAP@50: {checkpoint['best_map_50']:.4f}")
    return model

def predict_and_visualize_image_from_dataloader(model, data_loader, idx=None, device=torch.device('cuda'), confidence_threshold=0.5, figsize=(12, 12)):
    """
    Make prediction on a single image from a DataLoader and visualize results.
    """
    model.eval()

    # Flatten all batches into a list to index easily
    all_images = []
    all_targets = []

    for images, targets in data_loader:
        all_images.extend(images)
        all_targets.extend(targets)
    
    print(len(all_images))

    # Select random index if none provided
    if idx is None:
        idx = np.random.randint(0, len(all_images))

    # Get image and target
    image = all_images[idx]
    target = all_targets[idx]

    # Prepare image for model
    image_tensor = image.unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        prediction = model(image_tensor)

    # Convert image for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np * std + mean) * 255
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ground truth
    ax1.imshow(image_np)
    boxes = target['boxes'].cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
        ax1.add_patch(rect)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # Plot prediction
    ax2.imshow(image_np)
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    for box, score in zip(pred_boxes, pred_scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, y1 - 5, f'{score:.2f}', color='red')
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nGround Truth:")
    print(f"Number of objects: {len(boxes)}")
    print(f"\nPredictions (confidence > {confidence_threshold}):")
    print(f"Number of detections: {len(pred_scores[pred_scores > confidence_threshold])}")
    print("\nPrediction scores:", pred_scores[pred_scores > confidence_threshold])


def predict_and_visualize_image_dataset(model, test_dataset, idx=None, device = torch.device('cuda'), confidence_threshold=0.5, figsize=(12, 12)):
    """
    Make prediction on a single image and visualize results.
    """
    model.eval()

    # Select random index if none provided
    if idx is None:
        idx = np.random.randint(0, len(test_dataset))

    # Get image and target
    image, target = test_dataset[idx]

    # Prepare image for model
    image_tensor = image.unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        prediction = model(image_tensor)

    # Convert image for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np * std + mean) * 255
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot ground truth
    ax1.imshow(image_np)
    boxes = target['boxes'].cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
        ax1.add_patch(rect)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # Plot prediction
    ax2.imshow(image_np)
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()

    for box, score in zip(pred_boxes, pred_scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x1, y1 - 5, f'{score:.2f}', color='red')
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nGround Truth:")
    print(f"Number of objects: {len(boxes)}")
    print(f"\nPredictions (confidence > {confidence_threshold}):")
    print(f"Number of detections: {len(pred_scores[pred_scores > confidence_threshold])}")
    print("\nPrediction scores:", pred_scores[pred_scores > confidence_threshold])
