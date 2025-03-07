import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

class RumexDataSet(Dataset):
    def __init__(self, img_dir, annotation_dir, images_list, annotations_list, transform=None, augmentation_verbose=False, class_map=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.images_list = sorted(images_list)
        self.annotations_list = sorted(annotations_list)
        self.transform = transform
        self.augmentation_verbose = augmentation_verbose
        self.class_map = class_map if class_map else {}

    def __len__(self):
        return len(self.annotations_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images_list[idx])
        ann_path = os.path.join(self.annotation_dir, self.annotations_list[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        boxes = []
        labels = []
        for ann in annotation['annotations']:
            bbox = ann['bounding_box']
            x1 = bbox['x']
            y1 = bbox['y']
            x2 = x1 + bbox['w']
            y2 = y1 + bbox['h']
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map[ann['name']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes.numpy(), labels=labels.numpy())
            if self.augmentation_verbose:
                applied_transforms = [trans['__class_fullname__'] for trans in transformed["replay"]['transforms'] if trans['applied']]
                print(applied_transforms)
            image = transformed['image']
            if len(transformed['bboxes']) > 0:
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return image, target

def create_data_loaders(img_dir, annotation_dir, train_images, train_annotations, val_images, val_annotations, test_images, test_annotations, train_transform, valid_transform, batch_size=8, num_workers=2, class_map=None):
    train_dataset = RumexDataSet(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
        images_list=train_images,
        annotations_list=train_annotations,
        transform=train_transform,
        class_map=class_map
    )

    val_dataset = RumexDataSet(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
        images_list=val_images,
        annotations_list=val_annotations,
        transform=valid_transform,
        class_map=class_map
    )

    test_dataset = RumexDataSet(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
        images_list=test_images,
        annotations_list=test_annotations,
        transform=valid_transform,
        class_map=class_map
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )

    return train_loader, val_loader, test_loader