import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_sample(dataset, class_map, idx=None, figsize=(5,5)):
    """
    Visualize a sample from the dataset with proper normalization
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset))

    image, target = dataset[idx]

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    boxes = target['boxes'].numpy()
    labels = target['labels'].numpy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        for k, v in class_map.items():
            if v == label:
                class_name = k
                break
        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=1)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, class_name, color='red', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

    title = f"An image with the Train DataLoader\nNumber of objects: {len(boxes)}"
    ax.set_title(title)
    plt.axis('off')
    plt.show()

def visualize_augmentations(dataset_without_augmentation, dataset_with_augmentation, classes, idx=None, num_augmented=5, figsize=(15, 5)):
    """
    Visualize an original image side by side with its augmented versions.
    """
    if idx is None:
        idx = np.random.randint(0, len(dataset_without_augmentation))

    fig, axes = plt.subplots(1, num_augmented + 1, figsize=figsize)

    original_image, original_target = dataset_without_augmentation[idx]
    augmented_images = [dataset_with_augmentation[idx] for _ in range(num_augmented)]

    def plot_image(ax, image, target, title):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)

        ax.imshow(image)

        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            class_name = classes[label - 1]
            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, class_name, color='red', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

        ax.set_title(title)
        ax.axis('off')

    plot_image(axes[0], original_image, original_target, "Original Image")

    for i, (aug_image, aug_target) in enumerate(augmented_images):
        plot_image(axes[i + 1], aug_image, aug_target, f"Augmentation {i + 1}")

    plt.tight_layout()
    plt.show()

    print(f"\nImage index: {idx}")
    print(f"Number of augmented versions shown: {num_augmented}")
    print("Note: Each version may have different augmentations applied")