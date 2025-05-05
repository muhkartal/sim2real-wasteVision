import os
import sys
import time
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import airsim

logger = logging.getLogger('drone')

class Drone:

    def __init__(self, config: Dict[str, Any], client: airsim.MultirotorClient):
        self.config = config
        self.client = client
        self.drone_config = config['drone']
        self.image_config = config['image_generation']

        # Drone state
        self.current_position = None
        self.current_orientation = None
        self.current_altitude = None
        self.current_tilt = None

    def takeoff(self) -> None:
        self.client.takeoffAsync().join()
        logger.info("Drone takeoff completed")

    def set_random_position(self) -> Dict[str, Any]:
        altitude = random.uniform(
            self.drone_config['altitude_range'][0],
            self.drone_config['altitude_range'][1]
        )

        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        z = -altitude  # AirSim uses NED coordinate system (down is positive)

        self.client.moveToPositionAsync(x, y, z, 5).join()

        tilt_angle = random.uniform(
            self.drone_config['tilt_angle_range'][0],
            self.drone_config['tilt_angle_range'][1]
        )

        yaw = random.uniform(0, 360)

        self.client.rotateToYawAsync(yaw).join()

        pitch = -tilt_angle
        camera_pose = airsim.Pose(
            airsim.Vector3r(0, 0, 0),
            airsim.to_quaternion(pitch * np.pi / 180, 0, 0)
        )
        self.client.simSetCameraPose("high_res", camera_pose)

        # Get current state
        state = self.client.getMultirotorState()
        self.current_position = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])
        self.current_orientation = np.array([
            state.kinematics_estimated.orientation.w_val,
            state.kinematics_estimated.orientation.x_val,
            state.kinematics_estimated.orientation.y_val,
            state.kinematics_estimated.orientation.z_val
        ])
        self.current_altitude = altitude
        self.current_tilt = tilt_angle

        logger.info(f"Set drone position to [x={x:.2f}, y={y:.2f}, z={z:.2f}] "
                    f"with altitude={altitude:.2f}m, tilt={tilt_angle:.2f}°, yaw={yaw:.2f}°")

        return {
            'position': [x, y, z],
            'altitude': altitude,
            'tilt_angle': tilt_angle,
            'yaw': yaw
        }

    def capture_images(self) -> Dict[str, np.ndarray]:

        responses = self.client.simGetImages([
            airsim.ImageRequest("high_res", airsim.ImageType.Scene),  # RGB
            airsim.ImageRequest("high_res", airsim.ImageType.DepthVis),  # Depth visualization
            airsim.ImageRequest("high_res", airsim.ImageType.Segmentation)  # Segmentation
        ])

        rgb_response = responses[0]
        rgb_image = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        rgb_image = rgb_image.reshape(rgb_response.height, rgb_response.width, 3)

        depth_image = None
        if len(responses) > 1 and responses[1].pixels_as_float is not None:
            depth_response = responses[1]
            depth_image = np.array(depth_response.image_data_float, dtype=np.float32)
            depth_image = depth_image.reshape(depth_response.height, depth_response.width)

        segmentation_image = None
        if len(responses) > 2:
            seg_response = responses[2]
            segmentation_image = np.frombuffer(seg_response.image_data_uint8, dtype=np.uint8)
            segmentation_image = segmentation_image.reshape(seg_response.height, seg_response.width, 3)

        images = {'rgb': rgb_image}
        if depth_image is not None:
            images['depth'] = depth_image
        if segmentation_image is not None:
            images['segmentation'] = segmentation_image

        logger.info(f"Captured images with resolution {rgb_image.shape[1]}x{rgb_image.shape[0]}")

        return images

    def get_camera_info(self) -> Dict[str, Any]:

        camera_pose = self.client.simGetCameraPose("high_res")

        camera_info = self.client.simGetCameraInfo("high_res")

        return {
            'pose': {
                'position': [
                    camera_pose.position.x_val,
                    camera_pose.position.y_val,
                    camera_pose.position.z_val
                ],
                'orientation': [
                    camera_pose.orientation.w_val,
                    camera_pose.orientation.x_val,
                    camera_pose.orientation.y_val,
                    camera_pose.orientation.z_val
                ]
            },
            'fov': camera_info.fov,
            'resolution': [
                self.image_config['resolution']['width'],
                self.image_config['resolution']['height']
            ]
        }

    def get_camera_pose(self) -> airsim.Pose:
        return self.client.simGetCameraPose("high_res")

    def get_drone_state(self) -> Dict[str, Any]:

        return {
            'position': self.current_position.tolist() if self.current_position is not None else None,
            'orientation': self.current_orientation.tolist() if self.current_orientation is not None else None,
            'altitude': self.current_altitude,
            'tilt_angle': self.current_tilt
        }
