import yaml 
import os 
import json

def save_config(config, config_path):
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def read_yaml(file_path):
    """
    Read a YAML file and return its contents as a Python dictionary.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Contents of the YAML file
    """
    try:
        with open(file_path, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def format_darwin_related_pathes(dataset_name, darwin_root, dataset_version):
    """
    Format paths for the Darwin dataset
    """
    splits_dir  = os.path.join(darwin_root, dataset_name,"releases", dataset_version, "lists") 
    partition_name = os.listdir(splits_dir)[0]

    img_dir = os.path.join(darwin_root, dataset_name, "images")
    annotations_dir = os.path.join(darwin_root, dataset_name, "releases", dataset_version, "annotations")
    train_split_file = os.path.join(darwin_root, dataset_name, "releases", dataset_version, "lists", partition_name, "random_train.txt")
    test_split_file = os.path.join(darwin_root, dataset_name, "releases", dataset_version, "lists", partition_name, "random_test.txt")
    val_split_file = os.path.join(darwin_root, dataset_name, "releases", dataset_version, "lists", partition_name, "random_val.txt")

    n_train, s_test, s_val = partition_name.split("_")
    n_train = int(n_train)
    s_test = int(s_test)
    s_val = int(s_val)

    print(f"Train: {n_train}, Test: {s_test}, Val: {s_val}")

    return img_dir, annotations_dir, train_split_file, test_split_file, val_split_file
