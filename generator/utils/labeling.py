import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import cv2
import airsim

logger = logging.getLogger('labeling')

class LabelGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_width = config['image_generation']['resolution']['width']
        self.image_height = config['image_generation']['resolution']['height']
        self.label_format = config['rendering']['label_format']

        self.class_mapping = {}
        for i, waste_model in enumerate(config['waste']['models']):
            self.class_mapping[waste_model['name']] = i

    def generate_yolo_labels(self, ground_truth: List[Dict[str, Any]]) -> List[str]:
        yolo_labels = []

        for obj in ground_truth:
            class_id = obj['class_id']
            bbox = obj['bbox']  # [left, top, right, bottom]

            x_center = (bbox[0] + bbox[2]) / 2 / self.image_width
            y_center = (bbox[1] + bbox[3]) / 2 / self.image_height
            width = (bbox[2] - bbox[0]) / self.image_width
            height = (bbox[3] - bbox[1]) / self.image_height

            # Skip invalid boxes (e.g., too small or outside image)
            if width <= 0 or height <= 0 or x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                logger.warning(f"Skipping invalid bounding box for {obj['model']}: {bbox}")
                continue

            # Create YOLO format label
            yolo_label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_labels.append(yolo_label)

        return yolo_labels

    def generate_coco_labels(self, image_id: int, ground_truth: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotations = []

        for i, obj in enumerate(ground_truth):
            class_id = obj['class_id']
            bbox = obj['bbox']  # [left, top, right, bottom]

            coco_bbox = [
                bbox[0],  # left
                bbox[1],  # top
                bbox[2] - bbox[0],  # width
                bbox[3] - bbox[1]   # height
            ]

            if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                logger.warning(f"Skipping invalid bounding box for {obj['model']}: {bbox}")
                continue

            area = coco_bbox[2] * coco_bbox[3]

            annotation = {
                'id': i,
                'image_id': image_id,
                'category_id': class_id,
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0
            }

            annotations.append(annotation)

        return annotations

    def generate_labels(self, image_id: int, ground_truth: List[Dict[str, Any]]) -> Union[List[str], List[Dict[str, Any]]]:
        if self.label_format.lower() == 'yolo':
            return self.generate_yolo_labels(ground_truth)
        elif self.label_format.lower() == 'coco':
            return self.generate_coco_labels(image_id, ground_truth)
        else:
            logger.warning(f"Unsupported label format: {self.label_format}, defaulting to YOLO")
            return self.generate_yolo_labels(ground_truth)

    def save_yolo_label(self, label_path: Path, labels: List[str]) -> None:
        with open(label_path, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

        logger.debug(f"Saved YOLO labels to {label_path}")

    def save_coco_annotations(self, annotations: List[Dict[str, Any]], output_dir: Path) -> None:
        import json

        annotations_dir = output_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)

        annotations_file = annotations_dir / "instances.json"

        categories = []
        for model in self.config['waste']['models']:
            category = {
                'id': self.class_mapping[model['name']],
                'name': model['name'],
                'supercategory': 'waste'
            }
            categories.append(category)

        coco_data = {
            'info': {
                'description': 'Istanbul Waste Detection Dataset',
                'version': '1.0',
                'year': 2023,
                'contributor': 'Synthetic Data Generator',
                'date_created': '2023-01-01'
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'MIT License',
                    'url': 'https://opensource.org/licenses/MIT'
                }
            ],
            'images': [],  # Will be filled by the renderer
            'annotations': annotations,
            'categories': categories
        }

        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logger.info(f"Saved COCO annotations to {annotations_file}")

    def draw_annotations(self, image: np.ndarray, ground_truth: List[Dict[str, Any]]) -> np.ndarray:
        annotated_image = image.copy()

        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 128, 0),  # Teal
            (128, 0, 128)   # Purple
        ]

        for obj in ground_truth:
            class_id = obj['class_id']
            model_name = obj['model']
            bbox = [int(coord) for coord in obj['bbox']]  # [left, top, right, bottom]

            color = colors[class_id % len(colors)]

            cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            label = f"{model_name}"
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            cv2.rectangle(
                annotated_image,
                (bbox[0], bbox[1] - text_height - 5),
                (bbox[0] + text_width, bbox[1]),
                color,
                -1
            )

            cv2.putText(
                annotated_image,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )

        return annotated_image

    def get_class_info(self) -> Dict[str, Any]:

        classes = []
        for i, waste_model in enumerate(self.config['waste']['models']):
            class_info = {
                'id': i,
                'name': waste_model['name']
            }
            classes.append(class_info)

        return {
            'num_classes': len(classes),
            'classes': classes,
            'mapping': self.class_mapping
        }
