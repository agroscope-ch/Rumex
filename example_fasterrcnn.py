import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from torchabc import TorchABC
from functools import cached_property, partial
from typing import Any, Dict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from src.dataset import *
from utils.generic import *
from utils.data_inspection import *
from src.augmentation import *  


class ClassName(TorchABC):
    """A concrete implementation of the TorchABC abstract class.

    Use this template to implement your own model by following these steps:
    - replace ClassName with the name of your model,
    - replace this docstring with a description of your model,
    - implement the methods below to define the core logic of your model,
    - access the hyperparameters passed during initialization with `self.hparams`.
    """
    
    @cached_property
    def dataloaders(self):
        """The dataloaders for training and evaluation.

        This method defines and returns a dictionary containing the `DataLoader` instances
        for the training, validation, and testing datasets. The keys of the dictionary
        should correspond to the names of the datasets (e.g., 'train', 'val', 'test'),
        and the values should be their respective `torch.utils.data.DataLoader` objects.

        Any transformation of the raw input data for each dataset should be implemented
        within the `preprocess` method of this class. The `preprocess` method should 
        then be passed as the `transform` argument of the `Dataset` instances.

        If you require custom collation logic (i.e., a specific way to merge a list of
        samples into a batch beyond the default behavior), you should implement this
        logic in the `collate` method of this class. The `collate` method should then be 
        passed to the `collate_fn` argument when creating the `DataLoader` instances. 
        """
        config_file = "/home/naro/projects/Rumex/config/configs.yaml"
        config = read_yaml(config_file)

        dataset_name = config['dataset']['dataset_name']
        darwin_root = config['dataset']['darwin_root']
        dataset_version = config['dataset']['dataset_version']
        images_extension = config['dataset']['extension']


        img_dir, annotations_dir, train_split_file, test_split_file, val_split_file = format_darwin_related_pathes(dataset_name, darwin_root, dataset_version)

        # Initialize DataVerifier
        data_verifier = DataVerifier(
            img_dir = img_dir,
            annotations_dir = annotations_dir,
            train_split_file = train_split_file,
            test_split_file = test_split_file,
            val_split_file = val_split_file,
            extension = images_extension
        )

        # Verify data
        train_annotations, test_annotations, val_annotations = data_verifier.check_directory_contents()

        # Initialize ImageProcessor
        image_processor = ImagesClassesInspector(
            img_dir=img_dir,
            annotations_dir=annotations_dir
        )

        # Get image and annotation lists
        train_images = data_verifier.get_image_files(train_annotations)
        val_images = data_verifier.get_image_files(val_annotations)
        test_images = data_verifier.get_image_files(test_annotations)

        # Get image sizes
        image_files = os.listdir(img_dir)
        train_sizes = image_processor.get_image_sizes(image_files)

        # Get classes
        annotation_files = train_annotations + test_annotations + val_annotations
        classes = image_processor.get_classes(annotation_files)
        print("\nClasses in the dataset:")
        print(classes)

        class_map = {name: idx + 1 for idx, name in enumerate(classes)}
        print("\nThe created class map:")
        print(class_map)

        # Get image size stats
        min_size, max_size = image_processor.get_image_size_stats(image_files)
        print(f"Smallest image size: {min_size}")
        print(f"Largest image size: {max_size}")

        w_min, h_min = min_size
        print(f"Width of smallest image: {w_min}")
        print(f"Height of smallest image: {h_min}")

        augmentation_config = AugmentationConfig(height=h_min, width=w_min)

        # Get transforms
        train_transform = augmentation_config.get_train_transform()
        valid_transform = augmentation_config.get_valid_transform()



        train_dataloader, val_dataloader, test_dataloader = create_data_loaders(
            img_dir=img_dir,
            annotation_dir=annotations_dir,
            train_images=train_images,
            train_annotations=train_annotations,
            val_images=val_images,
            val_annotations=val_annotations,
            test_images=test_images,
            test_annotations=test_annotations,
            train_transform=train_transform,
            valid_transform=valid_transform,
            class_map=class_map,
            batch_size=8, 
            num_workers=2
        )


        return {'train': train_dataloader, 'val': val_dataloader}
    
    def preprocess(self, data: Any, flag: str = '') -> Any:
        """Prepare the raw data for the network.

        The way this method processes the `data` depends on the `flag`.
        When `flag` is empty (the default), the `data` are assumed to represent the 
        model's input that is used for inference. When `flag` has a specific value, 
        the method may perform different preprocessing steps such as transforming 
        the target or augmenting the input for training.

        Parameters
        ----------
        data : Any
            The raw input data to be processed.
        flag : str, optional
            A string indicating the purpose of the preprocessing. The default
            is an empty string, meaning preprocess the model's input for inference.

        Returns
        -------
        Any
            The preprocessed data.
        """
        transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.AtLeastOneBBoxRandomCrop(height=self.height, width=self.width, p=1),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.1),
            # A.Resize(height=self.height, width=self.width, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0),
        ], bbox_params=self.bbox_params)


        return transform(data)
    
    @cached_property
    def network(self):
        """The neural network.

        Returns a `torch.nn.Module` whose input and output tensors assume the
        batch size is the first dimension: (batch_size, ...).
        """
        num_classes = 2
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn( weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    
        # Replace the classifier with a new one for num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        return model
    
    @cached_property
    def optimizer(self):
        """The optimizer for training the network.

        Returns a `torch.optim.Optimizer` configured for `self.network.parameters()`.
        """
        return torch.optim.Adam(self.network.parameters(), lr=0.0001)

    
    @cached_property
    def scheduler(self):
        """The learning rate scheduler for the optimizer.

        Returns a `torch.optim.lr_scheduler.LRScheduler` or `torch.optim.lr_scheduler.ReduceLROnPlateau`
        configured for `self.optimizer`.
        """
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.1,
            patience=10,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=0,
            eps=1e-08)
    
    def loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Loss function.

        This method defines the loss function that quantifies the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        The loss function should be differentiable to enable backpropagation.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the computed loss value.
        """
        return outputs
    
    def metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Evaluation metrics.

        This method calculates various metrics that quantify the discrepancy
        between the neural network `outputs` and the corresponding `targets`. 
        Unlike `self.loss`, which is primarily used for training, these metrics 
        are only used for evaluation and they do not need to be differentiable.

        Parameters
        ----------
        outputs : torch.Tensor
            The tensor containing the network's output.
        targets : torch.Tensor
            The targets corresponding to the outputs.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are the names of the metrics and the 
            values are the corresponding metric scores.
        """
        return {}
    
    def postprocess(self, outputs: torch.Tensor) -> Any:
        """Postprocess the model's outputs.

        This method transforms the outputs of the neural network to 
        generate the final predictions. 

        Parameters
        ----------
        outputs : torch.Tensor
            The output tensor from `self.network`.

        Returns
        -------
        Any
            The postprocessed outputs.
        """
        return outputs
    

if __name__ == "__main__":
    # Example usage
    model = ClassName()
    model.train(epochs=1)
