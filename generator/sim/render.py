import os
import sys
import time
import json
import random
import logging
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool

import airsim

from generator.sim.environment import Environment
from generator.sim.waste_spawner import WasteSpawner
from generator.sim.drone import Drone
from generator.utils.lighting import LightingController
from generator.utils.labeling import LabelGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('renderer')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Synthetic image generator for waste detection')
    parser.add_argument('--config', type=str, default='config/sim_config.yaml',
                        help='Path to simulation configuration file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of images to generate (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated images with annotations')
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

def setup_output_dirs(output_dir: str) -> Tuple[Path, Path, Path]:
    output_path = Path(output_dir)

    # Create directories if they don't exist
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    viz_dir = output_path / "visualization"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directories setup at {output_path}")

    return output_path, images_dir, labels_dir

def generate_single_image(
    client: airsim.MultirotorClient,
    config: Dict[str, Any],
    env: Environment,
    waste_spawner: WasteSpawner,
    drone: Drone,
    lighting: LightingController,
    label_generator: LabelGenerator,
    image_id: int,
    output_path: Path,
    images_dir: Path,
    labels_dir: Path,
    visualize: bool = False
) -> Dict[str, Any]:
    try:
        env.reset()

        map_name = env.load_environment()

        lighting_params = lighting.set_lighting_condition()
        lighting.set_time_of_day()
        lighting.apply_post_processing()

        drone_params = drone.set_random_position()

        waste_objects = waste_spawner.spawn_waste_objects(
            np.array(drone_params['position'])
        )

        images = drone.capture_images()

        camera_pose = drone.get_camera_pose()
        camera_info = drone.get_camera_info()

        ground_truth = waste_spawner.get_object_ground_truth(
            camera_pose,
            config['image_generation']['resolution']['width'],
            config['image_generation']['resolution']['height']
        )
        labels = label_generator.generate_labels(image_id, ground_truth)

        image_filename = f"img_{image_id:06d}.jpg"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(images['rgb'], cv2.COLOR_RGB2BGR))

        if config['rendering']['label_format'].lower() == 'yolo':
            label_filename = f"img_{image_id:06d}.txt"
            label_path = labels_dir / label_filename
            label_generator.save_yolo_label(label_path, labels)

        if visualize:
            viz_dir = output_path / "visualization"
            viz_dir.mkdir(exist_ok=True)

            annotated_image = label_generator.draw_annotations(images['rgb'], ground_truth)

            viz_filename = f"viz_{image_id:06d}.jpg"
            viz_path = viz_dir / viz_filename
            cv2.imwrite(str(viz_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        metadata = {
            'image_id': image_id,
            'filename': image_filename,
            'resolution': camera_info['resolution'],
            'environment': map_name,
            'lighting': lighting_params,
            'drone': drone_params,
            'objects': [obj['model'] for obj in waste_objects],
            'num_objects': len(waste_objects),
            'class_distribution': {
                model['name']: len([obj for obj in waste_objects if obj['model'] == model['name']])
                for model in config['waste']['models']
            }
        }

        return metadata

    except Exception as e:
        logger.error(f"Error generating image {image_id}: {e}")
        return None

def worker_init(config_path: str):
    global g_config, g_client, g_env, g_waste_spawner, g_drone, g_lighting, g_label_generator

    g_config = load_config(config_path)

    g_client = airsim.MultirotorClient(
        ip=g_config['airsim']['host'],
        port=g_config['airsim']['port'],
        timeout_value=g_config['airsim']['timeout']
    )
    g_client.confirmConnection()

    g_env = Environment(g_config, g_client)
    g_waste_spawner = WasteSpawner(g_config, g_client)
    g_drone = Drone(g_config, g_client)
    g_lighting = LightingController(g_config, g_client)
    g_label_generator = LabelGenerator(g_config)

def worker_process(args):
    image_id, output_path, images_dir, labels_dir, visualize = args

    return generate_single_image(
        g_client, g_config, g_env, g_waste_spawner, g_drone, g_lighting, g_label_generator,
        image_id, Path(output_path), Path(images_dir), Path(labels_dir), visualize
    )

def generate_dataset(config: Dict[str, Any], args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else config['seed']
    random.seed(seed)
    np.random.seed(seed)

    output_dir = args.output_dir if args.output_dir else config['rendering']['output_dir']
    output_path, images_dir, labels_dir = setup_output_dirs(output_dir)

    num_images = args.num_images if args.num_images else config['image_generation']['total_images']
    logger.info(f"Generating {num_images} synthetic images")

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'simulation.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    client = airsim.MultirotorClient(
        ip=config['airsim']['host'],
        port=config['airsim']['port'],
        timeout_value=config['airsim']['timeout']
    )
    client.confirmConnection()

    env = Environment(config, client)
    waste_spawner = WasteSpawner(config, client)
    drone = Drone(config, client)
    lighting = LightingController(config, client)
    label_generator = LabelGenerator(config)

    drone.takeoff()

    all_metadata = []

    use_parallel = config['rendering']['num_workers'] > 1
    batch_size = config['rendering']['batch_size']
    num_workers = min(config['rendering']['num_workers'], mp.cpu_count())

    if use_parallel:
        logger.info(f"Using {num_workers} worker processes for parallel rendering")

        args_list = [
            (i, str(output_path), str(images_dir), str(labels_dir), args.visualize)
            for i in range(num_images)
        ]

        for batch_start in range(0, num_images, batch_size):
            batch_end = min(batch_start + batch_size, num_images)
            batch_args = args_list[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(num_images-1)//batch_size + 1} "
                        f"({batch_end-batch_start} images)")

            with Pool(num_workers, initializer=worker_init, initargs=(args.config,)) as pool:
                batch_results = list(tqdm(
                    pool.imap(worker_process, batch_args),
                    total=len(batch_args),
                    desc="Generating images"
                ))

            batch_metadata = [meta for meta in batch_results if meta is not None]
            all_metadata.extend(batch_metadata)

            logger.info(f"Completed batch with {len(batch_metadata)} successful images")
    else:
        logger.info("Using single process for rendering")

        with tqdm(total=num_images, desc="Generating images") as pbar:
            for i in range(num_images):
                metadata = generate_single_image(
                    client, config, env, waste_spawner, drone, lighting, label_generator,
                    i, output_path, images_dir, labels_dir, args.visualize
                )

                if metadata is not None:
                    all_metadata.append(metadata)

                pbar.update(1)

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'dataset_info': {
                'name': 'Istanbul Waste Detection Synthetic Dataset',
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'num_images': len(all_metadata),
                'image_format': 'jpg',
                'label_format': config['rendering']['label_format'],
                'resolution': [
                    config['image_generation']['resolution']['width'],
                    config['image_generation']['resolution']['height']
                ],
                'classes': [model['name'] for model in config['waste']['models']]
            },
            'generation_config': {
                'seed': seed,
                'environments': [map['name'] for map in config['unreal']['maps']],
                'lighting_conditions': [cond['name'] for cond in config['lighting']['conditions']]
            },
            'images': all_metadata
        }, f, indent=2)

    logger.info(f"Generated {len(all_metadata)} images with labels")
    logger.info(f"Metadata saved to {metadata_path}")

    if config['rendering']['label_format'].lower() == 'yolo':
        create_yolo_dataset_yaml(config, output_path)

def create_yolo_dataset_yaml(config: Dict[str, Any], output_path: Path) -> None:
    yolo_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: model['name'] for i, model in enumerate(config['waste']['models'])},
        'nc': len(config['waste']['models'])
    }

    yaml_path = output_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yolo_yaml, f, default_flow_style=False)

    logger.info(f"YOLO dataset configuration saved to {yaml_path}")

def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    log_level = getattr(logging, config['logging']['level'].upper())
    logger.setLevel(log_level)

    generate_dataset(config, args)

if __name__ == "__main__":
    main()
