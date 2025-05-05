import os
import json
import shutil
import argparse
import logging
import sys
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/setup_airsim.log', mode='w')
    ]
)
logger = logging.getLogger('setup_airsim')

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup AirSim for waste detection simulation')
    parser.add_argument('--config', type=str, default='config/sim_config.yaml',
                        help='Path to simulation configuration file')
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

def setup_airsim_dirs() -> None:
    """Create necessary AirSim directories."""
    home_dir = Path.home()
    airsim_settings_dir = home_dir / "Documents" / "AirSim"
    airsim_settings_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"AirSim settings directory: {airsim_settings_dir}")
    return airsim_settings_dir

def create_airsim_settings(config: Dict[str, Any],
                           settings_dir: Path) -> None:
    """Create AirSim settings JSON file."""
    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ViewMode": "",
        "ClockSpeed": 1.0,
        "Headless": config['airsim'].get('headless', False),

        "Vehicles": {
            "Drone1": {
                "VehicleType": "SimpleFlight",
                "DefaultVehicleState": "Armed",
                "EnableCollisionPassthrogh": False,
                "EnableCollisions": True,
                "AllowAPIAlways": True,
                "Cameras": {
                    "high_res": {
                        "CaptureSettings": [
                            {
                                "ImageType": 0,
                                "Width": config['image_generation']['resolution']['width'],
                                "Height": config['image_generation']['resolution']['height'],
                                "FOV_Degrees": 90,
                                "AutoExposureSpeed": 100,
                                "AutoExposureBias": 0,
                                "AutoExposureMaxBrightness": 0.64,
                                "AutoExposureMinBrightness": 0.03,
                                "MotionBlurAmount": 0,
                                "TargetGamma": 1.0,
                                "ProjectionMode": "",
                                "OrthoWidth": 5.12
                            }
                        ],
                        "X": 0.0,
                        "Y": 0.0,
                        "Z": 0.0,
                        "Pitch": 0.0,
                        "Roll": 0.0,
                        "Yaw": 0.0
                    }
                }
            }
        },

        "OriginGeopoint": {
            "Latitude": 41.0082,  # Istanbul latitude
            "Longitude": 28.9784,  # Istanbul longitude
            "Altitude": 0
        },

        "TimeOfDay": {
            "Enabled": True,
            "StartDateTime": "2022-05-05 12:00:00",
            "CelestialClock": {
                "Speed": 1,
                "StartDateTime": "2022-05-05 12:00:00"
            }
        },

        "SubWindows": [
            {"WindowID": 0, "CameraName": "high_res", "ImageType": 0,
             "VehicleName": "Drone1", "Visible": True}
        ]
    }

    # Add depth and segmentation settings if needed
    if 'depth' in config['image_generation']['formats']:
        depth_settings = {
            "ImageType": 2,  # Depth
            "Width": config['image_generation']['resolution']['width'],
            "Height": config['image_generation']['resolution']['height'],
            "FOV_Degrees": 90,
            "AutoExposureSpeed": 100,
            "AutoExposureBias": 0,
            "AutoExposureMaxBrightness": 0.64,
            "AutoExposureMinBrightness": 0.03,
            "MotionBlurAmount": 0,
            "TargetGamma": 1.0,
        }
        settings["Vehicles"]["Drone1"]["Cameras"]["high_res"]["CaptureSettings"].append(depth_settings)

    if 'segmentation' in config['image_generation']['formats']:
        segmentation_settings = {
            "ImageType": 5,  # Segmentation
            "Width": config['image_generation']['resolution']['width'],
            "Height": config['image_generation']['resolution']['height'],
            "FOV_Degrees": 90,
            "AutoExposureSpeed": 100,
            "AutoExposureBias": 0,
            "AutoExposureMaxBrightness": 0.64,
            "AutoExposureMinBrightness": 0.03,
            "MotionBlurAmount": 0,
            "TargetGamma": 1.0,
        }
        settings["Vehicles"]["Drone1"]["Cameras"]["high_res"]["CaptureSettings"].append(segmentation_settings)

    settings_path = settings_dir / "settings.json"
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=4)

    logger.info(f"Created AirSim settings at {settings_path}")

def check_airsim_installed() -> bool:
    """Check if AirSim is installed."""
    try:
        import airsim
        logger.info("AirSim Python package found")
        return True
    except ImportError:
        logger.error("AirSim Python package not found. Please install it using: pip install airsim")
        return False

def main() -> None:
    """Main function to set up AirSim."""
    os.makedirs('logs', exist_ok=True)

    args = parse_args()
    config = load_config(args.config)

    if not check_airsim_installed():
        sys.exit(1)

    airsim_settings_dir = setup_airsim_dirs()

    create_airsim_settings(config, airsim_settings_dir)

    logger.info("AirSim setup completed successfully")

if __name__ == "__main__":
    main()
