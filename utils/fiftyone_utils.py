import fiftyone as fo
import torch

def create_fiftyone_dataset(images, annotations, tags):
    """
    Create a FiftyOne dataset from images and annotations.

    Args:
        images (list): List of image file paths.
        annotations (list): List of annotation dictionaries.
        tags (list): List of tags for each image.

    Returns:
        fo.Dataset: FiftyOne dataset.
    """
    samples = []
    for img_path, ann, tag in zip(images, annotations, tags):
        sample = fo.Sample(filepath=img_path, tags=[tag])
        for box, label in zip(ann['boxes'], ann['labels']):
            det = fo.Detection(label=label, bounding_box=box)
            sample.ground_truth.detections.append(det)
        samples.append(sample)

    dataset = fo.Dataset(samples)
    return dataset

def add_predictions_to_fiftyone(dataset, model, device, confidence_threshold=0.5, tag="predictions"):
    """
    Add model predictions to a FiftyOne dataset.

    Args:
        dataset (fo.Dataset): FiftyOne dataset.
        model (torch.nn.Module): The model to use for inference.
        device (torch.device): The device to move the model and data to.
        confidence_threshold (float): The confidence threshold for filtering predictions.
        tag (str): The tag to use for the predictions.
    """
    model.eval()
    with torch.no_grad():
        for sample in dataset:
            image_path = sample.filepath
            image = fo.load_image(image_path)
            image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
            prediction = model(image_tensor)
            pred_boxes = prediction[0]['boxes'].cpu().numpy()
            pred_labels = prediction[0]['labels'].cpu().numpy()
            pred_scores = prediction[0]['scores'].cpu().numpy()

            detections = []
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if score > confidence_threshold:
                    det = fo.Detection(label=label, bounding_box=box, confidence=score)
                    detections.append(det)

            sample[tag] = fo.Detections(detections=detections)
            sample.save()

def evaluate_fiftyone_dataset(dataset, tag="predictions"):
    """
    Evaluate a FiftyOne dataset using all possible object detection metrics.

    Args:
        dataset (fo.Dataset): FiftyOne dataset.
        tag (str): The tag used for the predictions.
    """
    results = dataset.evaluate_detections(
        pred_field=tag,
        gt_field="ground_truth",
        eval_key="eval",
        compute_mAP=True,
        iou=0.5,
    )
    return results
