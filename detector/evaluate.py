import os
import sys
import logging
import argparse
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from ultralytics import YOLO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('evaluate')

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 waste detector')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to evaluation configuration file')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to evaluate (test, val)')
    parser.add_argument('--device', type=str, default=None,
                        help='Evaluation device (cpu, 0, 0,1,2,3, etc.)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save evaluation plots')
    parser.add_argument('--save-json', action='store_true',
                        help='Save evaluation results as JSON')
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

def load_dataset_config(data_path: str) -> Dict[str, Any]:
    """Load dataset configuration."""
    yaml_path = Path(data_path) / "data.yaml"

    if not yaml_path.exists():
        logger.error(f"Dataset configuration not found at {yaml_path}")
        sys.exit(1)

    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config
    except Exception as e:
        logger.error(f"Error loading dataset configuration: {e}")
        sys.exit(1)

def evaluate_model(
    model_path: str,
    data_path: str,
    split: str,
    config: Dict[str, Any],
    device: Optional[str] = None
) -> Dict[str, Any]:

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    data_config = load_dataset_config(data_path)

    # Determine path to test set
    if split == 'test':
        data_split_path = str(Path(data_path) / data_config['test'])
    elif split == 'val':
        data_split_path = str(Path(data_path) / data_config['val'])
    else:
        logger.error(f"Unknown split: {split}")
        sys.exit(1)

    logger.info(f"Evaluating model on {split} set: {data_split_path}")

    conf_thres = config['evaluation']['conf_thres']
    iou_thres = config['evaluation']['iou_thres']
    max_det = config['evaluation']['max_det']

    if device is None:
        device = 'cuda:0' if config.get('device') else 'cpu'

    logger.info(f"Running evaluation with conf_thres={conf_thres}, iou_thres={iou_thres}")
    results = model.val(
        data=data_split_path,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=device,
        verbose=True
    )

    metrics = {}

    if hasattr(results, 'box'):
        box_metrics = results.box

        metrics['overall'] = {
            'precision': float(box_metrics.p),  # Precision
            'recall': float(box_metrics.r),     # Recall
            'mAP50': float(box_metrics.map50),  # mAP@0.5
            'mAP50-95': float(box_metrics.map)  # mAP@0.5:0.95
        }

        if config['evaluation']['per_class'] and hasattr(box_metrics, 'cls_map50'):
            cls_names = list(data_config['names'].values())

            cls_metrics = []
            for i, cls_name in enumerate(cls_names):
                cls_metrics.append({
                    'class_id': i,
                    'class_name': cls_name,
                    'precision': float(box_metrics.cls_p[i]) if i < len(box_metrics.cls_p) else 0.0,
                    'recall': float(box_metrics.cls_r[i]) if i < len(box_metrics.cls_r) else 0.0,
                    'mAP50': float(box_metrics.cls_map50[i]) if i < len(box_metrics.cls_map50) else 0.0,
                    'mAP50-95': float(box_metrics.cls_map[i]) if i < len(box_metrics.cls_map) else 0.0
                })

            metrics['class'] = cls_metrics

    logger.info(f"Evaluation completed with mAP50={metrics['overall']['mAP50']:.4f}, "
                f"mAP50-95={metrics['overall']['mAP50-95']:.4f}")

    return metrics

def create_evaluation_report(
    metrics: Dict[str, Any],
    output_dir: Path,
    save_plots: bool = False,
    save_json: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_json:
        json_path = output_dir / "evaluation_metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation metrics saved to {json_path}")

    if 'class' in metrics:
        class_df = pd.DataFrame(metrics['class'])

        csv_path = output_dir / "class_metrics.csv"
        class_df.to_csv(csv_path, index=False)
        logger.info(f"Class metrics saved to {csv_path}")

        md_table = class_df.to_markdown(index=False, floatfmt=".4f")
        md_path = output_dir / "class_metrics.md"
        with open(md_path, 'w') as f:
            f.write("# YOLOv8 Waste Detector Evaluation Results\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Overall Metrics\n\n")
            f.write(f"- Precision: {metrics['overall']['precision']:.4f}\n")
            f.write(f"- Recall: {metrics['overall']['recall']:.4f}\n")
            f.write(f"- mAP@0.5: {metrics['overall']['mAP50']:.4f}\n")
            f.write(f"- mAP@0.5:0.95: {metrics['overall']['mAP50-95']:.4f}\n\n")
            f.write("## Class Metrics\n\n")
            f.write(md_table)
        logger.info(f"Markdown evaluation report saved to {md_path}")

        if save_plots:
            create_evaluation_plots(metrics, output_dir)
    else:
        logger.warning("No class metrics available for report")

def create_evaluation_plots(metrics: Dict[str, Any], output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Extract class metrics
    class_data = metrics['class']
    class_names = [c['class_name'] for c in class_data]
    precisions = [c['precision'] for c in class_data]
    recalls = [c['recall'] for c in class_data]
    map50s = [c['mAP50'] for c in class_data]
    map50_95s = [c['mAP50-95'] for c in class_data]

    plt.figure(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width/2, precisions, width, label='Precision')
    plt.bar(x + width/2, recalls, width, label='Recall')

    plt.xlabel('Waste Class')
    plt.ylabel('Score')
    plt.title('Precision and Recall by Waste Class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    pr_path = figures_dir / "precision_recall.png"
    plt.savefig(pr_path)
    plt.close()

    plt.figure(figsize=(12, 6))

    plt.bar(x - width/2, map50s, width, label='mAP@0.5')
    plt.bar(x + width/2, map50_95s, width, label='mAP@0.5:0.95')

    plt.xlabel('Waste Class')
    plt.ylabel('Score')
    plt.title('mAP Scores by Waste Class')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    map_path = figures_dir / "map_scores.png"
    plt.savefig(map_path)
    plt.close()

    logger.info(f"Evaluation plots saved to {figures_dir}")

def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    log_level = getattr(logging, config['logging']['level'].upper())
    logger.setLevel(log_level)

    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / 'evaluation.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    model_path = Path(args.weights)
    if not model_path.exists():
        logger.error(f"Model weights not found at {model_path}")
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        sys.exit(1)

    metrics = evaluate_model(
        model_path=str(model_path),
        data_path=str(data_path),
        split=args.split,
        config=config,
        device=args.device
    )

    output_dir = Path('docs') / 'evaluation'

    create_evaluation_report(
        metrics=metrics,
        output_dir=output_dir,
        save_plots=args.save_plots,
        save_json=args.save_json
    )

    logger.info("Evaluation completed successfully")

if __name__ == "__main__":
    main()
