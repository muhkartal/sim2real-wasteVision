import os
import sys
import logging
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
from datetime import datetime

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('train')

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 waste detector')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pre-trained weights (overrides config)')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Training device (cpu, 0, 0,1,2,3, etc.)')
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

def setup_output_dirs(config: Dict[str, Any]) -> Path:
    weights_dir = Path(config['output']['weights_dir'])
    weights_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['name']}_{timestamp}" if config['name'] else f"run_{timestamp}"

    return weights_dir

def train_model(config: Dict[str, Any], args: argparse.Namespace) -> Path:
    model_name = config['model']['name']
    pretrained = config['model']['pretrained']

    data_path = args.data if args.data else config['data']['yaml_path']

    epochs = args.epochs if args.epochs else config['hyperparameters']['epochs']
    batch_size = args.batch_size if args.batch_size else config['hyperparameters']['batch_size']
    image_size = config['hyperparameters']['image_size']
    device = args.device if args.device else 'cuda:0'

    if args.weights:
        logger.info(f"Loading pre-trained weights from {args.weights}")
        model = YOLO(args.weights)
    else:
        logger.info(f"Loading model {model_name}")
        model = YOLO(model_name)

    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset: {data_path}")

    train_args = {
        'data': data_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': image_size,
        'device': device,
        'project': config['project'],
        'name': config['name'],
        'patience': config['hyperparameters'].get('patience', 50),
        'lr0': config['hyperparameters'].get('lr0', 0.01),
        'lrf': config['hyperparameters'].get('lrf', 0.01),
        'momentum': config['hyperparameters'].get('momentum', 0.937),
        'weight_decay': config['hyperparameters'].get('weight_decay', 0.0005),
        'warmup_epochs': config['hyperparameters'].get('warmup_epochs', 3.0),
        'warmup_momentum': config['hyperparameters'].get('warmup_momentum', 0.8),
        'warmup_bias_lr': config['hyperparameters'].get('warmup_bias_lr', 0.1),
        'box': config['hyperparameters'].get('box', 7.5),
        'cls': config['hyperparameters'].get('cls', 0.5),
        'hsv_h': config['hyperparameters'].get('hsv_h', 0.015),
        'hsv_s': config['hyperparameters'].get('hsv_s', 0.7),
        'hsv_v': config['hyperparameters'].get('hsv_v', 0.4),
        'degrees': config['hyperparameters'].get('degrees', 0.0),
        'translate': config['hyperparameters'].get('translate', 0.1),
        'scale': config['hyperparameters'].get('scale', 0.5),
        'shear': config['hyperparameters'].get('shear', 0.0),
        'perspective': config['hyperparameters'].get('perspective', 0.0),
        'flipud': config['hyperparameters'].get('flipud', 0.0),
        'fliplr': config['hyperparameters'].get('fliplr', 0.5),
        'mosaic': config['hyperparameters'].get('mosaic', 1.0),
        'mixup': config['hyperparameters'].get('mixup', 0.0),
        'copy_paste': config['hyperparameters'].get('copy_paste', 0.0),
        'save_period': config['output'].get('save_period', -1),
        'exist_ok': config['output'].get('exist_ok', False),
        'verbose': config['output'].get('verbose', True),
        'save': config['output'].get('save', True)
    }

    start_time = time.time()

    results = model.train(**train_args)

    end_time = time.time()
    training_time = end_time - start_time

    logger.info(f"Training completed in {training_time:.2f} seconds")

    best_weights_path = Path(results.best) if hasattr(results, 'best') else None

    if best_weights_path and best_weights_path.exists():
        logger.info(f"Best model weights saved to {best_weights_path}")

        weights_dir = Path(config['output']['weights_dir'])
        target_path = weights_dir / "synthetic_only.pt"

        import shutil
        shutil.copy2(best_weights_path, target_path)
        logger.info(f"Copied best weights to {target_path}")

        return target_path
    else:
        logger.warning("Best model weights not found")
        return None

def save_training_results(results_path: Path, model_path: Path, config: Dict[str, Any]) -> None:
    if model_path is None:
        logger.warning("No model path provided, skipping results saving")
        return

    results = {
        'model': {
            'path': str(model_path),
            'name': config['model']['name'],
            'pretrained': config['model']['pretrained']
        },
        'training': {
            'epochs': config['hyperparameters']['epochs'],
            'batch_size': config['hyperparameters']['batch_size'],
            'image_size': config['hyperparameters']['image_size'],
            'optimizer': config['hyperparameters'].get('optimizer', 'SGD'),
            'learning_rate': config['hyperparameters'].get('lr0', 0.01)
        },
        'dataset': {
            'path': config['data']['path'],
            'classes': config['data']['classes']
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training results saved to {results_path}")

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

    weights_dir = setup_output_dirs(config)

    model_path = train_model(config, args)

    results_path = weights_dir / "training_results.json"
    save_training_results(results_path, model_path, config)

    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
