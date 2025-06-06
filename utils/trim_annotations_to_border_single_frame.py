import json
from typing import Dict, Any, Optional, Union, cast, List
import sys
import getopt
import os

class Trimmer:

    def trim(self, filename: str, output_filename: Optional[str]) -> None:
        with open(filename) as export_file:
            file_content: str = export_file.read()
            export: Dict[str, Any] = json.loads(file_content)
            
            image: Optional[Dict[str, Union[float, str]]] = export["item"]["slots"]
            if image is None:
                raise ValueError("No image found to convert.")

            max_width: float = cast(float, image[0].get('width'))
            max_height: float = cast(float, image[0].get('height'))
            annotations: List[Dict[str, Any]] = export.get("annotations", [])
            annotations_inside_image: List[Dict[str, Any]] = []
            for annotation in annotations:
                if not self._is_completely_outside_image(annotation, max_width, max_height):
                    annotations_inside_image.append(annotation)
            
            new_annotations: List[Dict[str, Any]] = []
            #print(annotations_inside_image)
            #print(new_annotations)
            for annotation in annotations_inside_image:
                new_annotations.append(self._trim_to_max_values(annotation, max_width, max_height))
                #print(new_annotations)
            new_export: Dict[str, Any] = export
            new_export["annotations"] = new_annotations
            
            if output_filename is None:
                old_filename, ext = os.path.splitext(filename)
                output_filename = f"{old_filename}_trimmed{ext}"

            with open(output_filename, "w") as trimmed_file:
                new_content = json.dumps(new_export)
                trimmed_file.write(new_content)
                trimmed_file.close()

    def _is_completely_outside_image(self, annotation: Dict[str, Any], max_width: float, max_height: float) -> bool:
        if self._is_bounding_box(annotation):
            bbox: Dict[str, float] = cast(Dict[str, float], annotation.get('bounding_box'))
            
            left_x: float = cast(float, bbox.get("x"))
            right_x: float = left_x + cast(float, bbox.get("w"))
            top_y: float = cast(float, bbox.get("y"))
            bottom_y: float = top_y + cast(float, bbox.get("h"))

            return (left_x < 0 and right_x < 0) or (left_x > max_width and right_x > max_width) or (top_y < 0 and bottom_y < 0) or (top_y > max_height and bottom_y > max_height)

        return False

    def _trim_to_max_values(self, annotation: Dict[str, Any], max_width: float, max_height: float) -> Dict[str, Any]:
        #print(annotation)
        #print(annotation.get('frames').get('0').get('polygon'))
        if self._is_bounding_box(annotation):
            bbox: Dict[str, float] = annotation.get('bounding_box', {})
            print(bbox)
            trimmed_bbox: Dict[str, float] = self._trim_bounding_box(bbox, max_width, max_height)
            #print(trimmed_bbox)
            annotation["bounding_box"] = trimmed_bbox
            #print(annotation["bounding_box"])
            #print(annotation)
            if self._is_polygon(annotation):
                polygon: Dict[str, float] = annotation.get('polygon', {})
                trimmed_polygon: Dict[str, float] = self._trim_polygon(polygon, max_width, max_height)
                annotation["polygon"] = trimmed_polygon
                #return annotation
            #return annotation
        print(annotation)
        return annotation

    def _is_bounding_box(self, annotation: Dict[str, Any]) -> bool:
        return annotation.get('bounding_box') is not None

    def _is_polygon(self, annotation: Dict[str, Any]) -> bool:
        return annotation.get('polygon') is not None

    def _trim_bounding_box(self, bbox: Dict[str, float], max_width: float, max_height: float) -> Dict[str, float]:
        left_x: float = cast(float, bbox.get("x"))
        top_y: float = cast(float, bbox.get("y"))
        width: float = cast(float, bbox.get("w"))
        height: float = cast(float, bbox.get("h"))

        if left_x < 0:
            left_x = 0
        
        if top_y < 0:
            top_y = 0

        if top_y > max_height:
            top_y = max_height
        
        if left_x > max_width:
            left_x = max_width

        right_x: float = left_x + width
        if right_x > max_width:
            x_diff: float = right_x - max_width
            width =  width - x_diff

        bottom_y = top_y + height
        if bottom_y > max_height:
            y_diff: float = bottom_y - max_height
            height = height - y_diff

        return {
            "h": height,
            "w": width,
            "x": left_x,
            "y": top_y
        }

    def _trim_polygon(self, polygon: Dict[str, float], max_width: float, max_height: float) -> Dict[str, float]:
        
        length_x = len(polygon.get('paths')[0])
        #print(type(polygon.get('paths')[0][0]))
        new_list = []
        final_list = [] 
        for i in range(length_x):
            left_x: float = cast(float, polygon.get('paths')[0][i].get("x"))
            top_y: float = cast(float, polygon.get('paths')[0][i].get("y"))
            if left_x < 0:
                left_x = 0
        
            if top_y < 0:
                top_y = 0

            if top_y > max_height:
                top_y = max_height
        
            if left_x > max_width:
                left_x = max_width

            dicts = {"x": left_x,"y": top_y}
            new_list.append(dicts)

        final_list.append(new_list)
        parent_dict = {'paths': final_list}
        #return {
        #    "x": left_x,
        #    "y": top_y
        #}
        return parent_dict

def main(argv):
    # Error codes
    FILENAME_MISSING_ERROR_CODE: int = 1
    BAD_OPTIONS_ERROR_CODE: int = 2

    # Messages
    HELP_MESSAGE: str = "Usage: test.py -i <inputfile> -o <outputfile>\nWhere -o is optional and will default to <inputfile>_trimmed."
    SUCCESS_MESSAGE: str = "Operation completed successfully!"

    if len(argv) <= 1:
        print(HELP_MESSAGE)
        sys.exit(FILENAME_MISSING_ERROR_CODE)

    input_file: str = ""
    output_file: Optional[str] = None
   
    try:
        opts, args = getopt.getopt(argv[1:], "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print(HELP_MESSAGE)
        sys.exit(BAD_OPTIONS_ERROR_CODE)
    
    for opt, arg in opts:
        if opt == "-h":
            print(HELP_MESSAGE)
            sys.exit()
        elif opt in ("-i"):
            input_file = arg
        elif opt in ("-o"):
            output_file = arg

    trimmer = Trimmer()
    trimmer.trim(input_file, output_file)

    print(SUCCESS_MESSAGE) 

if __name__ == "__main__":
   main(sys.argv)