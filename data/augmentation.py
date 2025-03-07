import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial
import cv2

class AugmentationConfig:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=0,
            min_visibility=0.2,  # Adjust as needed
        )

    def get_train_transform(self):
        """
        Training transforms with augmentations
        """
        return A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.AtLeastOneBBoxRandomCrop(height=1024, width=678, p=1),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
            A.Resize(height=self.height, width=self.width, always_apply=True),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0),
        ], bbox_params=self.bbox_params)

    def get_valid_transform(self):
        """
        Validation transforms without augmentations
        """
        return A.Compose([
            A.Resize(height=self.height, width=self.width, always_apply=True),  # Resize first
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0),
        ], bbox_params=self.bbox_params)

# Example usage
if __name__ == "__main__":
    from config.paths_config import PathsConfig
    from data.data_inspection import ImagesClassesInspector

    paths_config = PathsConfig(
        dataset_name="haldennord09",
        darwin_root="/home/naro/.darwin/datasets/digital-production",
        partition_name="2700_386_772",
        dataset_version="latest",
        extension='png',
        models_dir='/home/naro/projects/Rumex/models'
    )

    image_processor = ImagesClassesInspector(
        img_dir=paths_config.img_dir,
        annotations_dir=paths_config.annotations_dir
    )

    image_files = os.listdir(paths_config.img_dir)
    min_size, max_size = image_processor.get_image_size_stats(image_files)
    w_min, h_min = min_size

    augmentation_config = AugmentationConfig(height=h_min, width=w_min)

    train_transform = augmentation_config.get_train_transform()
    valid_transform = augmentation_config.get_valid_transform()

    print("Training transforms:")
    print(train_transform)
    print("\nValidation transforms:")
    print(valid_transform)
