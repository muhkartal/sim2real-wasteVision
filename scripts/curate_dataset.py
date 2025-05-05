import os
import sys
import json
import random
import logging
import argparse
import yaml
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('dataset_curation')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Curate synthetic dataset for waste detection')
    parser.add_argument('--config', type=str, default='config/dataset_config.yaml',
                        help='Path to dataset configuration file')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def load_metadata(input_dir: Path) -> Dict[str, Any]:
    metadata_path = input_dir / "metadata.json"
    logger.info(f"Loading metadata from {metadata_path}")

    if not metadata_path.exists():
        logger.error(f"Metadata file not found at {metadata_path}")
        sys.exit(1)

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        sys.exit(1)

def create_split_directories(output_dir: Path, format: str) -> Dict[str, Path]:
    split_dirs = {}

    for split in ['train', 'val', 'test']:
        if format.lower() == 'yolo':
            split_img_dir = output_dir / 'images' / split
            split_label_dir = output_dir / 'labels' / split
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = {
                'images': split_img_dir,
                'labels': split_label_dir
            }
        elif format.lower() == 'coco':
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            split_dirs[split] = split_dir

    logger.info(f"Created dataset split directories for format: {format}")
    return split_dirs

def validate_yolo_label(label_path: Path) -> Tuple[bool, int]:

    if not label_path.exists():
        return False, 0

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return False, 0

        valid_lines = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                # Check if values are within valid range
                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 < width <= 1 and 0 < height <= 1):
                    valid_lines += 1
            except ValueError:
                continue

        return valid_lines > 0, valid_lines
    except Exception:
        return False, 0

def split_dataset(
    metadata: Dict[str, Any],
    input_dir: Path,
    split_dirs: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, List[str]]:

    images = metadata['images']

    random.shuffle(images)

    train_ratio = config['splits']['train']
    val_ratio = config['splits']['val']
    test_ratio = config['splits']['test']

    num_images = len(images)
    train_size = int(num_images * train_ratio)
    val_size = int(num_images * val_ratio)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    logger.info(f"Split dataset: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    split_mapping = {
        'train': [img['image_id'] for img in train_images],
        'val': [img['image_id'] for img in val_images],
        'test': [img['image_id'] for img in test_images]
    }

    return split_mapping

def copy_dataset_files(
    input_dir: Path,
    split_dirs: Dict[str, Any],
    split_mapping: Dict[str, List[str]],
    format: str
) -> Dict[str, Dict[str, int]]:

    stats = {
        'train': {'images': 0, 'labels': 0, 'objects_per_class': defaultdict(int)},
        'val': {'images': 0, 'labels': 0, 'objects_per_class': defaultdict(int)},
        'test': {'images': 0, 'labels': 0, 'objects_per_class': defaultdict(int)}
    }

    if format.lower() == 'yolo':
        for split, image_ids in split_mapping.items():
            logger.info(f"Copying {len(image_ids)} images to {split} set")

            for image_id in tqdm(image_ids, desc=f"Copying {split} set"):
                image_file = f"img_{image_id:06d}.jpg"
                label_file = f"img_{image_id:06d}.txt"

                image_src = input_dir / "images" / image_file
                label_src = input_dir / "labels" / label_file

                image_dst = split_dirs[split]['images'] / image_file
                label_dst = split_dirs[split]['labels'] / label_file

                if not image_src.exists():
                    logger.warning(f"Image file not found: {image_src}")
                    continue

                shutil.copy2(image_src, image_dst)
                stats[split]['images'] += 1

                is_valid, num_objects = validate_yolo_label(label_src)
                if is_valid:
                    shutil.copy2(label_src, label_dst)
                    stats[split]['labels'] += 1

                    with open(label_src, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                class_name = config['classes'][class_id]['name']
                                stats[split]['objects_per_class'][class_name] += 1
                else:
                    logger.warning(f"Invalid label file: {label_src}")

    elif format.lower() == 'coco':
        logger.info("COCO format handling not fully implemented")

    logger.info("Dataset files copied to split directories")
    return stats

def create_yolo_dataset_yaml(output_dir: Path, config: Dict[str, Any]) -> None:
    names = {c['id']: c['name'] for c in config['classes']}

    yolo_config = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(config['classes']),
        'names': names
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_config, f, default_flow_style=False)

    logger.info(f"Created YOLO dataset configuration at {yaml_path}")

def generate_dataset_metadata(
    output_dir: Path,
    original_metadata: Dict[str, Any],
    split_mapping: Dict[str, List[str]],
    stats: Dict[str, Dict[str, int]],
    config: Dict[str, Any]
) -> None:
    dataset_info = {
        'name': 'Istanbul Waste Detection Synthetic Dataset',
        'description': 'Synthetic dataset for waste detection from drone imagery in Istanbul',
        'version': '1.0',
        'format': config['format'],
        'total_images': sum(len(ids) for ids in split_mapping.values()),
        'splits': {
            'train': len(split_mapping['train']),
            'val': len(split_mapping['val']),
            'test': len(split_mapping['test'])
        },
        'classes': [
            {
                'id': c['id'],
                'name': c['name'],
                'count': {
                    'train': stats['train']['objects_per_class'].get(c['name'], 0),
                    'val': stats['val']['objects_per_class'].get(c['name'], 0),
                    'test': stats['test']['objects_per_class'].get(c['name'], 0),
                    'total': (
                        stats['train']['objects_per_class'].get(c['name'], 0) +
                        stats['val']['objects_per_class'].get(c['name'], 0) +
                        stats['test']['objects_per_class'].get(c['name'], 0)
                    )
                }
            }
            for c in config['classes']
        ],
        'statistics': {
            'environments': {},
            'lighting_conditions': {},
            'altitude_ranges': {
                '0-5m': 0,
                '5-10m': 0,
                '10-15m': 0
            },
            'tilt_angle_ranges': {
                '0-10°': 0,
                '10-20°': 0,
                '20-30°': 0
            },
            'objects_per_image': {
                '1-5': 0,
                '6-10': 0,
                '11-15': 0
            }
        },
        'generation_parameters': original_metadata.get('generation_config', {})
    }

    image_stats = dataset_info['statistics']
    env_stats = image_stats['environments']
    light_stats = image_stats['lighting_conditions']

    for img in original_metadata['images']:
        env = img.get('environment')
        if env:
            env_stats[env] = env_stats.get(env, 0) + 1

        light = img.get('lighting', {}).get('condition')
        if light:
            light_stats[light] = light_stats.get(light, 0) + 1

        altitude = img.get('drone', {}).get('altitude')
        if altitude is not None:
            if 0 <= altitude < 5:
                image_stats['altitude_ranges']['0-5m'] += 1
            elif 5 <= altitude < 10:
                image_stats['altitude_ranges']['5-10m'] += 1
            elif 10 <= altitude <= 15:
                image_stats['altitude_ranges']['10-15m'] += 1

        tilt = img.get('drone', {}).get('tilt_angle')
        if tilt is not None:
            if 0 <= tilt < 10:
                image_stats['tilt_angle_ranges']['0-10°'] += 1
            elif 10 <= tilt < 20:
                image_stats['tilt_angle_ranges']['10-20°'] += 1
            elif 20 <= tilt <= 30:
                image_stats['tilt_angle_ranges']['20-30°'] += 1

        num_objects = img.get('num_objects', 0)
        if 1 <= num_objects <= 5:
            image_stats['objects_per_image']['1-5'] += 1
        elif 6 <= num_objects <= 10:
            image_stats['objects_per_image']['6-10'] += 1
        elif 11 <= num_objects <= 15:
            image_stats['objects_per_image']['11-15'] += 1

    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    logger.info(f"Generated detailed dataset metadata at {metadata_path}")

def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    log_level = getattr(logging, config['logging']['level'].upper())
    logger.setLevel(log_level)

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / config['logging']['file'], mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    metadata = load_metadata(input_dir)

    split_dirs = create_split_directories(output_dir, config['format'])

    split_mapping = split_dataset(metadata, input_dir, split_dirs, config)

    stats = copy_dataset_files(input_dir, split_dirs, split_mapping, config['format'])

    if config['format'].lower() == 'yolo':
        create_yolo_dataset_yaml(output_dir, config)

    generate_dataset_metadata(output_dir, metadata, split_mapping, stats, config)

    logger.info("Dataset curation completed successfully")

if __name__ == "__main__":
    main()
