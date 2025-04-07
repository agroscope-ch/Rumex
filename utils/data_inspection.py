import os
import sys
from PIL import Image
import json

class DataVerifier:
    def __init__(self, img_dir, annotations_dir, train_split_file, test_split_file, val_split_file, extension):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.train_split_file = train_split_file
        self.test_split_file = test_split_file
        self.val_split_file = val_split_file
        self.extension = extension

    def check_directory_contents(self):
        # Check contents of image and annotations directories
        print("Checking directory contents...")
        print("\nFirst few images in image directory:")
        print(os.listdir(self.img_dir)[:5])

        print("\nFirst few annotations in annotation directory:")
        print(os.listdir(self.annotations_dir)[:5])

        # Read and print first few lines of split files that have been created with the darwin API
        print("\nFirst few lines in split files:")
        print("Train split:")
        with open(self.train_split_file, 'r') as f:
            train_annotations = [os.path.basename(line.strip()) for line in f.readlines()]
            print(train_annotations[:5])

        print("\nTest split:")
        with open(self.test_split_file, 'r') as f:
            test_annotations = [os.path.basename(line.strip()) for line in f.readlines()]
            print(test_annotations[:5])

        print("\nVal split:")
        with open(self.val_split_file, 'r') as f:
            val_annotations = [os.path.basename(line.strip()) for line in f.readlines()]
            print(val_annotations[:5])

        return train_annotations, test_annotations, val_annotations

    def get_image_files(self, annotation_files):
        return [f.replace('.json', f'.{self.extension}') for f in annotation_files]

    def verify_file_existence(self, train_images, train_annotations):
        print("\nVerifying file existence...")
        for i in range(3):  # Check first 3 files
            if i < len(train_images):
                img_path = os.path.join(self.img_dir, train_images[i])
                ann_path = os.path.join(self.annotations_dir, train_annotations[i])
                print(f"\nChecking training pair {i+1}:")
                print(f"Image exists: {os.path.exists(img_path)} - {train_images[i]}")
                print(f"Annotation exists: {os.path.exists(ann_path)} - {train_annotations[i]}")


class ImagesClassesInspector:
    def __init__(self, img_dir, annotations_dir):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir

    def get_image_sizes(self, image_files):
        sizes = []
        for img_file in image_files:
            im_path = os.path.join(self.img_dir, img_file)
            img = Image.open(im_path)
            sizes.append(img.size)
        return sizes

    def get_classes(self, annotation_files):
        classes = []
        for ann_file in annotation_files:
            with open(os.path.join(self.annotations_dir, ann_file), 'r') as f:
                ann = json.load(f)
                for obj in ann['annotations']:
                    if obj['name'] not in classes:
                        classes.append(obj['name'])
        return classes

    def get_image_size_stats(self, image_files):
        sizes = self.get_image_sizes(image_files)
        min_size = min(sizes)
        max_size = max(sizes)
        return min_size, max_size

# Example usage
if __name__ == "__main__":
    from config.paths_config import *
    from data.data_inspection import *

    paths_config = PathsConfig(
        dataset_name="haldennord09",
        darwin_root="/home/naro/.darwin/datasets/digital-production",
        partition_name="2700_386_772",
        dataset_version="latest",
        extension='png',
        models_dir='/home/naro/projects/Rumex/models'
    )

    data_verifier = DataVerifier(
        img_dir=paths_config.img_dir,
        annotations_dir=paths_config.annotations_dir,
        train_split_file=paths_config.train_split_file,
        test_split_file=paths_config.test_split_file,
        val_split_file=paths_config.val_split_file,
        extension=paths_config.extension
    )

    train_annotations, test_annotations, val_annotations = data_verifier.check_directory_contents()

    image_processor = ImagesClassesInspector(
        img_dir=paths_config.img_dir,
        annotations_dir=paths_config.annotations_dir
    )

    image_files = os.listdir(paths_config.img_dir)
    train_sizes = image_processor.get_image_sizes(image_files)

    annotation_files = train_annotations + test_annotations + val_annotations
    classes = image_processor.get_classes(annotation_files)
    print("\nClasses in the dataset:")
    print(classes)

    class_map = {name: idx + 1 for idx, name in enumerate(classes)}
    print("\nThe created class map:")
    print(class_map)

    min_size, max_size = image_processor.get_image_size_stats(image_files)
    print(f"Smallest image size: {min_size}")
    print(f"Largest image size: {max_size}")

    w_min, h_min = min_size
    print(f"Width of smallest image: {w_min}")
    print(f"Height of smallest image: {h_min}")
