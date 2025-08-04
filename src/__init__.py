"""
UAV Strategic Deconfliction System

A comprehensive system for detecting and analyzing conflicts in shared airspace
between multiple drone missions operating simultaneously.

FlytBase Robotics Assignment 2025
"""

__version__ = "1.0.0"
__author__ = "FlytBase Robotics Team"

from .drone_mission import DroneMission, Waypoint
from .deconfliction_service import DeconflictionService, ConflictDetector
from .visualization import Visualizer
from .data_generator import DataGenerator
from .utils import setup_logging, save_results, load_results

__all__ = [
    'DroneMission',
    'Waypoint', 
    'DeconflictionService',
    'ConflictDetector',
    'Visualizer',
    'DataGenerator',
    'setup_logging',
    'save_results',
    'load_results'
]