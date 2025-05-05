import os
import sys
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import airsim

logger = logging.getLogger('waste_spawner')

class WasteSpawner:


    def __init__(self, config: Dict[str, Any], client: airsim.MultirotorClient):
        self.config = config
        self.client = client
        self.waste_models = config['waste']['models']
        self.density_range = config['waste']['density']
        self.clustering = config['waste']['clustering']

        self.spawned_objects = []

    def _get_random_waste_model(self) -> Dict[str, Any]:
        model_weights = [m['proportion'] for m in self.waste_models]
        return random.choices(self.waste_models, weights=model_weights, k=1)[0]

    def _get_spawn_locations(self, num_objects: int,
                            area_center: np.ndarray,
                            area_size: np.ndarray) -> List[np.ndarray]:
        locations = []

        use_clustering = (self.clustering['enabled'] and
                         random.random() < self.clustering['probability'])

        if use_clustering:
            num_clusters = max(1, min(num_objects // 3, 5))  # Between 1 and 5 clusters
            cluster_centers = []

            for _ in range(num_clusters):
                x = area_center[0] + random.uniform(-area_size[0]/2, area_size[0]/2)
                y = area_center[1] + random.uniform(-area_size[1]/2, area_size[1]/2)
                z = area_center[2]  # Keep z coordinate constant (ground level)
                cluster_centers.append(np.array([x, y, z]))

            objects_per_cluster = [num_objects // num_clusters] * num_clusters
            for i in range(num_objects % num_clusters):
                objects_per_cluster[i] += 1

            for cluster_idx, center in enumerate(cluster_centers):
                cluster_size = random.uniform(
                    self.clustering['cluster_radius_range'][0],
                    self.clustering['cluster_radius_range'][1]
                )

                for _ in range(objects_per_cluster[cluster_idx]):
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.uniform(0, cluster_size)

                    x = center[0] + radius * np.cos(angle)
                    y = center[1] + radius * np.sin(angle)
                    z = center[2]

                    locations.append(np.array([x, y, z]))
        else:
            for _ in range(num_objects):
                x = area_center[0] + random.uniform(-area_size[0]/2, area_size[0]/2)
                y = area_center[1] + random.uniform(-area_size[1]/2, area_size[1]/2)
                z = area_center[2]  # Keep z coordinate constant (ground level)
                locations.append(np.array([x, y, z]))

        return locations

    def spawn_waste_objects(self, drone_position: np.ndarray) -> List[Dict[str, Any]]:
        self.clear_objects()

        num_objects = random.randint(
            self.density_range['min'],
            self.density_range['max']
        )

        logger.info(f"Spawning {num_objects} waste objects")

        heading = 0
        spawn_center = drone_position + np.array([10 * np.cos(heading), 10 * np.sin(heading), 0])
        spawn_size = np.array([20, 20])

        spawn_locations = self._get_spawn_locations(num_objects, spawn_center, spawn_size)

        spawned_objects = []

        for i, location in enumerate(spawn_locations):
            model = self._get_random_waste_model()

            scale = random.uniform(model['scale_range'][0], model['scale_range'][1])

            rotation_z = random.uniform(model['rotation_range'][0], model['rotation_range'][1])

            object_name = f"{model['name']}_{i}"

            object_info = {
                'name': object_name,
                'model': model['name'],
                'position': location.tolist(),
                'rotation': [0, 0, rotation_z],
                'scale': scale,
                'class_id': next((i for i, m in enumerate(self.waste_models)
                                if m['name'] == model['name']), 0)
            }

            spawned_objects.append(object_info)

        self.spawned_objects = spawned_objects
        logger.info(f"Spawned {len(spawned_objects)} waste objects")

        return spawned_objects

    def clear_objects(self) -> None:
        """
        Clear all spawned waste objects.
        """
        if not self.spawned_objects:
            return

        self.spawned_objects = []
        logger.info("Cleared all waste objects")

    def get_objects_in_view(self, camera_pose: airsim.Pose) -> List[Dict[str, Any]]:


        return self.spawned_objects

    def get_object_ground_truth(self, camera_pose: airsim.Pose,
                               image_width: int,
                               image_height: int) -> List[Dict[str, Any]]:

        ground_truth = []

        for obj in self.spawned_objects:
            center_x = random.uniform(0.2, 0.8) * image_width
            center_y = random.uniform(0.2, 0.8) * image_height

            width = random.uniform(30, 100) * obj['scale']
            height = random.uniform(30, 100) * obj['scale']

            left = max(0, center_x - width/2)
            top = max(0, center_y - height/2)
            right = min(image_width, center_x + width/2)
            bottom = min(image_height, center_y + height/2)

            bbox = [left, top, right, bottom]

            ground_truth.append({
                'name': obj['name'],
                'model': obj['model'],
                'class_id': obj['class_id'],
                'bbox': bbox,
                'position': obj['position']
            })

        return ground_truth
