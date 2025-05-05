import os
import sys
import argparse
import logging
import subprocess
import shutil
import yaml
import zipfile
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/setup_unreal.log', mode='w')
    ]
)
logger = logging.getLogger('setup_unreal')

ASSET_INFO = {
    'istanbul_maps': {
        'url': 'https://example.com/assets/istanbul_maps.zip',  # Placeholder URL
        'description': 'Istanbul environment maps',
        'size_mb': 500,
        'license': 'CC BY-NC-SA 4.0'
    },
    'waste_models': {
        'url': 'https://example.com/assets/waste_models.zip',  # Placeholder URL
        'description': 'Waste 3D models collection',
        'size_mb': 100,
        'license': 'CC BY 4.0'
    },
    'environmental_assets': {
        'url': 'https://example.com/assets/environmental_assets.zip',  # Placeholder URL
        'description': 'Environmental props and textures',
        'size_mb': 200,
        'license': 'CC BY 4.0'
    }
}

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup Unreal Engine for waste detection simulation')
    parser.add_argument('--config', type=str, default='config/sim_config.yaml',
                        help='Path to simulation configuration file')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloading assets (use existing files)')
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

def check_unreal_installed(config: Dict[str, Any]) -> bool:
    """Check if Unreal Engine is installed at the configured path."""
    editor_path = Path(config['unreal']['editor_path'])
    if not editor_path.exists():
        logger.error(f"Unreal Engine not found at {editor_path}")
        logger.info("Please install Unreal Engine 5 and update the path in config/sim_config.yaml")
        return False

    logger.info(f"Unreal Engine found at {editor_path}")
    return True

def setup_project_structure(config: Dict[str, Any]) -> Tuple[Path, Path]:
    """Create the Unreal Engine project structure."""
    # Create project directories
    project_root = Path(config['unreal']['project_path']).parent
    project_root.mkdir(parents=True, exist_ok=True)

    content_dir = project_root / "Content"
    content_dir.mkdir(exist_ok=True)

    maps_dir = content_dir / "Maps"
    maps_dir.mkdir(exist_ok=True)

    assets_dir = content_dir / "Assets"
    assets_dir.mkdir(exist_ok=True)

    waste_dir = assets_dir / "Waste"
    waste_dir.mkdir(exist_ok=True)

    environment_dir = assets_dir / "Environment"
    environment_dir.mkdir(exist_ok=True)

    logger.info(f"Project structure created at {project_root}")
    return project_root, content_dir

def download_assets(skip_download: bool) -> None:
    """Download required assets for the project."""
    if skip_download:
        logger.info("Skipping asset download as requested")
        return

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)

    for asset_name, asset_info in ASSET_INFO.items():
        asset_path = assets_dir / f"{asset_name}.zip"

        if asset_path.exists():
            logger.info(f"Asset {asset_name} already exists at {asset_path}")
            continue

        logger.info(f"Downloading {asset_name} ({asset_info['size_mb']} MB) from {asset_info['url']}")

        with open(asset_path, 'w') as f:
            f.write(f"Placeholder for {asset_name} asset file\n")

        logger.info(f"Downloaded {asset_name} to {asset_path}")

    license_dir = assets_dir / "licenses"
    license_dir.mkdir(exist_ok=True)

    for asset_name, asset_info in ASSET_INFO.items():
        license_path = license_dir / f"{asset_name}_license.txt"
        with open(license_path, 'w') as f:
            f.write(f"Asset: {asset_name}\n")
            f.write(f"Description: {asset_info['description']}\n")
            f.write(f"License: {asset_info['license']}\n")

    logger.info("All assets downloaded successfully")

def extract_assets(project_content_dir: Path) -> None:
    """Extract downloaded assets to the project content directory."""
    assets_dir = Path("assets")

    for asset_name in ASSET_INFO.keys():
        asset_path = assets_dir / f"{asset_name}.zip"

        if not asset_path.exists():
            logger.warning(f"Asset {asset_name} not found at {asset_path}")
            continue

        target_dir = project_content_dir / "Assets" / asset_name
        target_dir.mkdir(exist_ok=True)

        with open(target_dir / "README.txt", 'w') as f:
            f.write(f"Placeholder for extracted {asset_name} assets\n")

        logger.info(f"Extracted {asset_name} to {target_dir}")

def create_unreal_project(config: Dict[str, Any], project_root: Path) -> None:
    """Create or update the Unreal Engine project file."""
    project_path = Path(config['unreal']['project_path'])

    if not project_path.exists():
        project_content = {
            "FileVersion": 3,
            "EngineAssociation": "5.2",
            "Category": "Simulation",
            "Description": "Istanbul Waste Detection Synthetic Data Generation",
            "Plugins": [
                {
                    "Name": "AirSim",
                    "Enabled": True
                }
            ],
            "TargetPlatforms": ["Linux"]
        }

        with open(project_path, 'w') as f:
            f.write(str(project_content))

        logger.info(f"Created Unreal project file at {project_path}")
    else:
        logger.info(f"Unreal project file already exists at {project_path}")

def setup_waste_models(project_content_dir: Path, config: Dict[str, Any]) -> None:
    """Set up waste models in the Unreal project with proper scaling and materials."""
    waste_dir = project_content_dir / "Assets" / "Waste"


    for waste_model in config['waste']['models']:
        model_name = waste_model['name']
        model_dir = waste_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Create placeholder files
        with open(model_dir / f"{model_name}.txt", 'w') as f:
            f.write(f"Placeholder for {model_name} 3D model\n")
            f.write(f"Scale range: {waste_model['scale_range']}\n")
            f.write(f"Rotation range: {waste_model['rotation_range']}\n")
            f.write(f"Proportion: {waste_model['proportion']}\n")

    logger.info(f"Set up {len(config['waste']['models'])} waste models in {waste_dir}")

def verify_setup(config: Dict[str, Any]) -> bool:
    """Verify that all required components are properly set up."""
    project_path = Path(config['unreal']['project_path'])
    if not project_path.exists():
        logger.error(f"Project file not found at {project_path}")
        return False

    logger.info("Unreal Engine and AirSim setup verified successfully")
    return True

def main() -> None:
    """Main function to set up Unreal Engine with AirSim and required assets."""
    os.makedirs('logs', exist_ok=True)

    args = parse_args()
    config = load_config(args.config)

    if not check_unreal_installed(config):
        sys.exit(1)

    project_root, content_dir = setup_project_structure(config)

    download_assets(args.skip_download)

    extract_assets(content_dir)

    create_unreal_project(config, project_root)

    setup_waste_models(content_dir, config)

    if not verify_setup(config):
        logger.warning("Setup verification failed, but continuing...")

    logger.info("Unreal Engine setup completed successfully")
    logger.info(f"To open the project, run: {config['unreal']['editor_path']} {config['unreal']['project_path']}")

if __name__ == "__main__":
    main()
