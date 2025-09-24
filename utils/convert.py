from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional

class YOLOToCOCOConverter:
    def __init__(
        self,
        yolo_root: str,
        output_dir: str,
        class_names: List[str],
        image_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    ):
        """
        Args:
            yolo_root: Path to YOLO dataset (with train/valid subdirectories)
            output_dir: Where to save COCO JSON files
            class_names: List of class names (index matches YOLO class_ids)
            image_extensions: Supported image formats
        """
        self.yolo_path = Path(yolo_root)
        self.output_path = Path(output_dir)
        self.class_names = class_names
        self.image_extensions = image_extensions
        
        # Validate paths
        if not self.yolo_path.exists():
            raise FileNotFoundError(f"YOLO directory not found: {self.yolo_path}")
        
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _convert_split(self, split: str) -> Optional[Path]:
        """Convert a single split (train/valid) to COCO format."""
        image_dir = self.yolo_path / split / "images"
        label_dir = self.yolo_path / split / "labels"
        
        if not image_dir.exists():
            print(f"Warning: {image_dir} does not exist, skipping")
            return None

        # Initialize COCO dataset structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": i, "name": name} 
                for i, name in enumerate(self.class_names)
            ]
        }

        annotation_id = 0
        image_paths = [
            p for p in image_dir.iterdir() 
            if p.suffix.lower() in self.image_extensions
        ]

        for img_id, img_path in enumerate(tqdm(image_paths, desc=f"Processing {split}")):
            try:
                # Read image with Pillow
                with Image.open(img_path) as img:
                    width, height = img.size

                # Add image info
                coco_data["images"].append({
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                })

                # Parse corresponding label file
                label_path = label_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue

                        class_id, x_center, y_center, w, h = map(float, parts)
                        class_id = int(class_id)

                        # Convert YOLO to COCO bbox format
                        x = (x_center - w/2) * width
                        y = (y_center - h/2) * height
                        w *= width
                        h *= height

                        # Add annotation
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": class_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

        # Save COCO JSON
        output_json = self.output_path / f"instances_{split}.json"
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        return output_json

    def convert(self) -> Dict[str, Path]:
        """Convert both train and valid splits."""
        results = {}
        for split in ["train", "valid"]:
            output_path = self._convert_split(split)
            if output_path:
                results[split] = output_path
        return results


if __name__ == "__main__":
    # Example Usage
    converter = YOLOToCOCOConverter(
        yolo_root="dcm_data",
        output_dir="coco_output",
        class_names=["class1", "class2", "class3"]  # Replace with your class names
    )
    
    result_paths = converter.convert()
    print(f"Conversion complete. Output files: {result_paths}")