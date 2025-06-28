import json
import os
import sys
import getopt
from typing import Dict, Any, Optional, Union, cast, List
from tqdm import tqdm

class Trimmer:

    def trim(self, input_path: str, output_path: Optional[str] = None) -> None:
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path '{input_path}' is not a directory.")

        json_files = [f for f in os.listdir(input_path) if f.endswith('.json')]
        if not json_files:
            print("No JSON files found in the input directory.")
            return

        for filename in tqdm(json_files):
            input_file = os.path.join(input_path, filename)
            output_file = (
                os.path.join(output_path, filename) if output_path
                else os.path.splitext(input_file)[0] + "_trimmed.json"
            )
            self._process_file(input_file, output_file)

    def _process_file(self, filename: str, output_filename: str) -> None:
        with open(filename) as export_file:
            export: Dict[str, Any] = json.load(export_file)

            image = export["item"].get("slots")
            if image is None:
                raise ValueError("No image found to convert.")

            max_width = cast(float, image[0].get('width'))
            max_height = cast(float, image[0].get('height'))
            annotations = export.get("annotations", [])

            annotations_inside = [
                a for a in annotations if not self._is_completely_outside_image(a, max_width, max_height)
            ]
            trimmed_annotations = [self._trim_to_max_values(a, max_width, max_height) for a in annotations_inside]
            export["annotations"] = trimmed_annotations

            with open(output_filename, "w") as out_file:
                json.dump(export, out_file)

    def _is_completely_outside_image(self, annotation: Dict[str, Any], max_width: float, max_height: float) -> bool:
        if not self._is_bounding_box(annotation):
            return False

        bbox = cast(Dict[str, float], annotation['bounding_box'])
        x, y = bbox.get("x", 0), bbox.get("y", 0)
        w, h = bbox.get("w", 0), bbox.get("h", 0)

        return (x + w < 0 or x > max_width) or (y + h < 0 or y > max_height)

    def _trim_to_max_values(self, annotation: Dict[str, Any], max_width: float, max_height: float) -> Dict[str, Any]:
        if self._is_bounding_box(annotation):
            annotation["bounding_box"] = self._trim_bounding_box(annotation["bounding_box"], max_width, max_height)
            if self._is_polygon(annotation):
                annotation["polygon"] = self._trim_polygon(annotation["polygon"], max_width, max_height)
        return annotation

    def _is_bounding_box(self, annotation: Dict[str, Any]) -> bool:
        return 'bounding_box' in annotation

    def _is_polygon(self, annotation: Dict[str, Any]) -> bool:
        return 'polygon' in annotation

    def _trim_bounding_box(self, bbox: Dict[str, float], max_width: float, max_height: float) -> Dict[str, float]:
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        x, y = max(0, min(x, max_width)), max(0, min(y, max_height))
        w = max(0, min(w, max_width - x))
        h = max(0, min(h, max_height - y))
        return {"x": x, "y": y, "w": w, "h": h}

    def _trim_polygon(self, polygon: Dict[str, float], max_width: float, max_height: float) -> Dict[str, float]:
        new_paths = []
        for pt in polygon.get('paths', [[]])[0]:
            x, y = pt["x"], pt["y"]
            new_paths.append({"x": max(0, min(x, max_width)), "y": max(0, min(y, max_height))})
        return {"paths": [new_paths]}

def main(argv):
    HELP = "Usage: python script.py -i <input_folder> [-o <output_folder>]"
    if len(argv) <= 1:
        print(HELP)
        sys.exit(1)

    input_dir, output_dir = "", None

    try:
        opts, _ = getopt.getopt(argv[1:], "hi:o:", ["ifolder=", "ofolder="])
    except getopt.GetoptError:
        print(HELP)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(HELP)
            sys.exit()
        elif opt in ("-i", "--ifolder"):
            input_dir = arg
        elif opt in ("-o", "--ofolder"):
            output_dir = arg

    if not input_dir:
        print("Input folder is required.")
        print(HELP)
        sys.exit(1)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trimmer = Trimmer()
    trimmer.trim(input_dir, output_dir)
    print("Operation completed successfully!")

if __name__ == "__main__":
    main(sys.argv)
