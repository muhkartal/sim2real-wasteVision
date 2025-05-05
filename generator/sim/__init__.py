"""
Simulation modules for Istanbul Waste Detection.

This package contains modules for setting up and controlling
the AirSim simulation environment.

SPDX-License-Identifier: MIT
"""

from .environment import Environment
from .waste_spawner import WasteSpawner
from .drone import Drone
from .render import generate_dataset
