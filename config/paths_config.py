import os

class PathsConfig:
    def __init__(self, dataset_name, darwin_root, dataset_version, extension, models_dir):
        self.dataset_name = dataset_name
        self.darwin_root = darwin_root
        self.dataset_version = dataset_version
        self.extension = extension
        self.models_dir = models_dir

        splits_dir  = os.path.join(self.darwin_root, self.dataset_name,"releases", self.dataset_version, "lists") 
        self.partition_name = os.listdir(splits_dir)[0]

        self.img_dir = os.path.join(self.darwin_root, self.dataset_name, "images")
        self.annotations_dir = os.path.join(self.darwin_root, self.dataset_name, "releases", self.dataset_version, "annotations")
        self.train_split_file = os.path.join(self.darwin_root, self.dataset_name, "releases", self.dataset_version, "lists", self.partition_name, "random_train.txt")
        self.test_split_file = os.path.join(self.darwin_root, self.dataset_name, "releases", self.dataset_version, "lists", self.partition_name, "random_test.txt")
        self.val_split_file = os.path.join(self.darwin_root, self.dataset_name, "releases", self.dataset_version, "lists", self.partition_name, "random_val.txt")

        n_train, s_test, s_val = self.partition_name.split("_")
        n_train = int(n_train)
        s_test = int(s_test)
        s_val = int(s_val)

        print(f"Train: {n_train}, Test: {s_test}, Val: {s_val}")


        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)