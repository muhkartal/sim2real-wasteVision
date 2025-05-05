import os
import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import airsim

logger = logging.getLogger('lighting')

class LightingController:

    def __init__(self, config: Dict[str, Any], client: airsim.MultirotorClient):

        self.config = config
        self.client = client
        self.lighting_config = config['lighting']

        self.current_condition = None

    def set_lighting_condition(self, condition: Optional[str] = None) -> Dict[str, Any]:
        lighting_conditions = self.lighting_config['conditions']

        if condition is None:
            condition_weights = [cond['proportion'] for cond in lighting_conditions]
            condition_names = [cond['name'] for cond in lighting_conditions]
            condition = random.choices(condition_names, weights=condition_weights, k=1)[0]

        selected_condition = None
        for cond in lighting_conditions:
            if cond['name'] == condition:
                selected_condition = cond
                break

        if selected_condition is None:
            logger.warning(f"Lighting condition '{condition}' not found, using default")
            selected_condition = lighting_conditions[0]

        intensity = random.uniform(
            selected_condition['intensity_range'][0],
            selected_condition['intensity_range'][1]
        )

        shadow_strength = random.uniform(
            selected_condition['shadow_strength_range'][0],
            selected_condition['shadow_strength_range'][1]
        )

        color_temp = None
        if 'color_temperature_range' in selected_condition:
            color_temp = random.uniform(
                selected_condition['color_temperature_range'][0],
                selected_condition['color_temperature_range'][1]
            )

        logger.info(f"Setting lighting condition: {condition} "
                    f"(intensity={intensity:.2f}, shadow={shadow_strength:.2f})")


        lighting_params = {
            'condition': condition,
            'intensity': intensity,
            'shadow_strength': shadow_strength
        }

        if color_temp is not None:
            lighting_params['color_temperature'] = color_temp

        self.current_condition = lighting_params

        return lighting_params

    def set_time_of_day(self, time_str: Optional[str] = None) -> None:

        condition_to_time = {
            'Sunny': '12:00',
            'Overcast': '15:00',
            'Dawn': '06:30',
            'Dusk': '19:30'
        }

        if time_str is None and self.current_condition is not None:
            condition = self.current_condition['condition']
            time_str = condition_to_time.get(condition, '12:00')

        if time_str is None:
            time_str = '12:00'

        logger.info(f"Setting time of day to {time_str}")

    def apply_post_processing(self) -> None:
        if self.current_condition is None:
            return

        condition = self.current_condition['condition']

        post_processing_params = {
            'Sunny': {
                'bloom': 1.0,
                'auto_exposure_bias': 0.0,
                'gamma': 1.0
            },
            'Overcast': {
                'bloom': 0.8,
                'auto_exposure_bias': -0.5,
                'gamma': 1.1
            },
            'Dawn': {
                'bloom': 1.5,
                'auto_exposure_bias': -1.0,
                'gamma': 0.9
            },
            'Dusk': {
                'bloom': 1.5,
                'auto_exposure_bias': -1.0,
                'gamma': 0.9
            }
        }

        params = post_processing_params.get(condition, post_processing_params['Sunny'])

        logger.info(f"Applying post-processing for {condition}: {params}")


    def get_lighting_state(self) -> Dict[str, Any]:
        return self.current_condition
