import os
import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import airsim

logger = logging.getLogger('environment')

class Environment:

    def __init__(self, config: Dict[str, Any], client: airsim.MultirotorClient = None):

        self.config = config
        self.client = client if client else self._connect_client()
        self.current_map = None
        self.maps = config['unreal']['maps']

        self.lighting_condition = None

    def _connect_client(self) -> airsim.MultirotorClient:

        client = airsim.MultirotorClient(
            ip=self.config['airsim']['host'],
            port=self.config['airsim']['port'],
            timeout_value=self.config['airsim']['timeout']
        )
        client.confirmConnection()
        logger.info("Connected to AirSim client")
        return client

    def load_environment(self, map_name: Optional[str] = None) -> str:
        if map_name is None:
            map_weights = [m['proportion'] for m in self.maps]
            map_names = [m['name'] for m in self.maps]
            map_name = random.choices(map_names, weights=map_weights, k=1)[0]

        logger.info(f"Loading environment map: {map_name}")

        self.current_map = map_name


        return map_name

    def set_lighting_condition(self, condition: Optional[str] = None) -> Dict[str, Any]:
        lighting_config = self.config['lighting']['conditions']

        if condition is None:
            condition_weights = [c['proportion'] for c in lighting_config]
            condition_names = [c['name'] for c in lighting_config]
            condition = random.choices(condition_names, weights=condition_weights, k=1)[0]

        selected_lighting = None
        for cond in lighting_config:
            if cond['name'] == condition:
                selected_lighting = cond
                break

        if selected_lighting is None:
            logger.warning(f"Lighting condition '{condition}' not found, using default")
            selected_lighting = lighting_config[0]

        intensity = random.uniform(
            selected_lighting['intensity_range'][0],
            selected_lighting['intensity_range'][1]
        )

        shadow_strength = random.uniform(
            selected_lighting['shadow_strength_range'][0],
            selected_lighting['shadow_strength_range'][1]
        )

        color_temp = None
        if 'color_temperature_range' in selected_lighting:
            color_temp = random.uniform(
                selected_lighting['color_temperature_range'][0],
                selected_lighting['color_temperature_range'][1]
            )

        logger.info(f"Setting lighting condition: {condition} "
                    f"(intensity={intensity:.2f}, shadow={shadow_strength:.2f})")

        lighting_params = {
            'condition': condition,
            'intensity': intensity,
            'shadow_strength': shadow_strength
        }

        if color_temp:
            lighting_params['color_temperature'] = color_temp

        self.lighting_condition = lighting_params

        return lighting_params

    def reset(self) -> None:
        """
        Reset the environment to initial state.
        """
        self.client.reset()

        self.client.enableApiControl(True)

        self.client.armDisarm(True)

        logger.info("Environment reset")

    def get_environment_state(self) -> Dict[str, Any]:
        """
        Get the current environment state.

        Returns:
            Dictionary with environment state
        """
        return {
            'map': self.current_map,
            'lighting': self.lighting_condition
        }

    def close(self) -> None:
        """
        Close the environment and release resources.
        """
        self.client.armDisarm(False)

        self.client.enableApiControl(False)

        logger.info("Environment closed")
