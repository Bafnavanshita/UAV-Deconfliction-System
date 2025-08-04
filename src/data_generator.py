"""
Data Generator Module

This module provides utilities for generating realistic test data
for the UAV deconfliction system, including random missions and scenarios.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Any
from .drone_mission import DroneMission, Waypoint

class DataGenerator:
    """Generates realistic drone mission data for testing and demonstration."""
    
    def __init__(self, seed: int = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Define airspace boundaries
        self.airspace_bounds = {
            'x_min': 0, 'x_max': 500,
            'y_min': 0, 'y_max': 500,
            'z_min': 5, 'z_max': 150
        }
        
        # Realistic drone parameters
        self.drone_types = {
            'delivery': {
                'speed_range': (5, 15),  # m/s
                'altitude_range': (10, 50),
                'safety_buffer': 5.0
            },
            'surveillance': {
                'speed_range': (3, 8),
                'altitude_range': (30, 100),
                'safety_buffer': 8.0
            },
            'inspection': {
                'speed_range': (2, 6),
                'altitude_range': (5, 30),
                'safety_buffer': 3.0
            },
            'emergency': {
                'speed_range': (10, 25),
                'altitude_range': (20, 120),
                'safety_buffer': 10.0
            }
        }
    
    def generate_random_waypoint(self, altitude_range: Tuple[float, float] = None) -> Tuple[float, float, float]:
        """
        Generate a random waypoint within airspace boundaries.
        
        Args:
            altitude_range: Optional altitude range override
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        x = np.random.uniform(self.airspace_bounds['x_min'], self.airspace_bounds['x_max'])
        y = np.random.uniform(self.airspace_bounds['y_min'], self.airspace_bounds['y_max'])
        
        if altitude_range:
            z = np.random.uniform(altitude_range[0], altitude_range[1])
        else:
            z = np.random.uniform(self.airspace_bounds['z_min'], self.airspace_bounds['z_max'])
        
        return (x, y, z)
    
    def generate_mission_waypoints(self, drone_type: str, num_waypoints: int = None) -> List[Waypoint]:
        """
        Generate realistic waypoints for a specific drone type.
        
        Args:
            drone_type: Type of drone ('delivery', 'surveillance', etc.)
            num_waypoints: Number of waypoints to generate
            
        Returns:
            List of waypoints with timing information
        """
        if drone_type not in self.drone_types:
            drone_type = 'delivery'  # Default fallback
        
        drone_config = self.drone_types[drone_type]
        
        if num_waypoints is None:
            num_waypoints = np.random.randint(3, 8)
        
        waypoints = []
        current_time = 0.0
        
        # Generate waypoints with realistic movement patterns
        for i in range(num_waypoints):
            if i == 0:
                # Starting position
                position = self.generate_random_waypoint(drone_config['altitude_range'])
            else:
                # Generate next position based on previous waypoint and drone capabilities
                prev_waypoint = waypoints[-1]
                
                # Calculate realistic next position
                max_distance = np.random.uniform(30, 150)  # Reasonable segment distance
                angle = np.random.uniform(0, 2 * np.pi)
                
                new_x = prev_waypoint.x + max_distance * np.cos(angle)
                new_y = prev_waypoint.y + max_distance * np.sin(angle)
                
                # Keep within bounds
                new_x = np.clip(new_x, self.airspace_bounds['x_min'], self.airspace_bounds['x_max'])
                new_y = np.clip(new_y, self.airspace_bounds['y_min'], self.airspace_bounds['y_max'])
                
                # Altitude variation
                altitude_change = np.random.uniform(-20, 20)
                new_z = prev_waypoint.z + altitude_change
                new_z = np.clip(new_z, drone_config['altitude_range'][0], drone_config['altitude_range'][1])
                
                position = (new_x, new_y, new_z)
                
                # Calculate travel time based on distance and speed
                distance = np.sqrt((new_x - prev_waypoint.x)**2 + 
                                 (new_y - prev_waypoint.y)**2 + 
                                 (new_z - prev_waypoint.z)**2)
                
                speed = np.random.uniform(*drone_config['speed_range'])
                travel_time = distance / speed
                current_time += travel_time
            
            waypoint = Waypoint(position[0], position[1], position[2], current_time)
            waypoints.append(waypoint)
        
        return waypoints
    
    def generate_random_mission(self, drone_id: str, drone_type: str = None, 
                              start_time: float = None, mission_duration: float = None) -> DroneMission:
        """
        Generate a complete random drone mission.
        
        Args:
            drone_id: Unique identifier for the drone
            drone_type: Type of drone mission
            start_time: Mission start time (random if None)
            mission_duration: Mission duration (calculated if None)
            
        Returns:
            Complete DroneMission object
        """
        if drone_type is None:
            drone_type = np.random.choice(list(self.drone_types.keys()))
        
        drone_config = self.drone_types[drone_type]
        
        # Generate waypoints
        num_waypoints = np.random.randint(3, 7)
        waypoints = self.generate_mission_waypoints(drone_type, num_waypoints)
        
        # Set mission timing
        if start_time is None:
            start_time = np.random.uniform(0, 100)
        
        # Adjust waypoint times to start from mission start time
        if waypoints:
            time_offset = start_time - waypoints[0].time
            for waypoint in waypoints:
                waypoint.time += time_offset
        
        if mission_duration is None:
            mission_duration = waypoints[-1].time - waypoints[0].time if waypoints else 60
        
        end_time = start_time + mission_duration
        
        return DroneMission(
            drone_id=drone_id,
            waypoints=waypoints,
            start_time=start_time,
            end_time=end_time,
            safety_buffer=drone_config['safety_buffer']
        )
    
    def generate_multiple_missions(self, count: int, id_prefix: str = "DRONE") -> List[DroneMission]:
        """
        Generate multiple random missions.
        
        Args:
            count: Number of missions to generate
            id_prefix: Prefix for drone IDs
            
        Returns:
            List of DroneMission objects
        """
        missions = []
        
        for i in range(count):
            drone_id = f"{id_prefix}_{i+1:03d}"
            drone_type = np.random.choice(list(self.drone_types.keys()))
            
            mission = self.generate_random_mission(drone_id, drone_type)
            missions.append(mission)
        
        return missions
    
    def generate_conflict_scenario(self, primary_drone_id: str = "PRIMARY_001") -> Tuple[DroneMission, List[DroneMission]]:
        """
        Generate a scenario specifically designed to create conflicts.
        
        Args:
            primary_drone_id: ID for the primary mission
            
        Returns:
            Tuple of (primary_mission, list_of_conflicting_missions)
        """
        # Generate primary mission
        primary_mission = self.generate_random_mission(primary_drone_id, 'delivery')
        
        # Generate conflicting missions
        conflicting_missions = []
        
        # Spatial conflict - mission that crosses primary path
        spatial_conflict = self._generate_spatial_conflict(primary_mission, "CONFLICT_SPATIAL_001")
        conflicting_missions.append(spatial_conflict)
        
        # Temporal conflict - mission in same area at overlapping time
        temporal_conflict = self._generate_temporal_conflict(primary_mission, "CONFLICT_TEMPORAL_001")
        conflicting_missions.append(temporal_conflict)
        
        # Near-miss scenario
        near_miss = self._generate_near_miss(primary_mission, "NEAR_MISS_001")
        conflicting_missions.append(near_miss)
        
        return primary_mission, conflicting_missions
    
    def _generate_spatial_conflict(self, primary_mission: DroneMission, conflict_id: str) -> DroneMission:
        """Generate a mission that spatially conflicts with the primary mission."""
        # Pick a waypoint from primary mission to conflict with
        target_waypoint = np.random.choice(primary_mission.waypoints[1:-1])  # Avoid start/end
        
        # Create waypoints that pass through or near the target area
        conflict_waypoints = []
        
        # Start point away from conflict area
        start_x = target_waypoint.x + np.random.uniform(-100, 100)
        start_y = target_waypoint.y + np.random.uniform(-100, 100)
        start_z = target_waypoint.z + np.random.uniform(-10, 10)
        
        start_time = target_waypoint.time - np.random.uniform(5, 15)
        conflict_waypoints.append(Waypoint(start_x, start_y, start_z, start_time))
        
        # Conflict point (very close to target waypoint)
        conflict_x = target_waypoint.x + np.random.uniform(-3, 3)
        conflict_y = target_waypoint.y + np.random.uniform(-3, 3)
        conflict_z = target_waypoint.z + np.random.uniform(-2, 2)
        
        conflict_time = target_waypoint.time + np.random.uniform(-2, 2)
        conflict_waypoints.append(Waypoint(conflict_x, conflict_y, conflict_z, conflict_time))
        
        # End point
        end_x = conflict_x + np.random.uniform(-50, 50)
        end_y = conflict_y + np.random.uniform(-50, 50)
        end_z = conflict_z + np.random.uniform(-10, 10)
        
        end_time = conflict_time + np.random.uniform(10, 20)
        conflict_waypoints.append(Waypoint(end_x, end_y, end_z, end_time))
        
        return DroneMission(
            drone_id=conflict_id,
            waypoints=conflict_waypoints,
            start_time=start_time,
            end_time=end_time,
            safety_buffer=5.0
        )
    
    def _generate_temporal_conflict(self, primary_mission: DroneMission, conflict_id: str) -> DroneMission:
        """Generate a mission that temporally conflicts with the primary mission."""
        # Find overlapping time window
        overlap_start = primary_mission.start_time + np.random.uniform(0, 30)
        overlap_end = overlap_start + np.random.uniform(20, 60)
        
        # Pick a location near primary mission path
        primary_trajectory = primary_mission.get_trajectory_points(1.0)
        if primary_trajectory:
            reference_point = np.random.choice(primary_trajectory)
            base_x, base_y, base_z = reference_point[0], reference_point[1], reference_point[2]
        else:
            base_x, base_y, base_z = 100, 100, 30
        
        # Create waypoints in the same general area
        conflict_waypoints = []
        
        # Generate waypoints with small spatial offset but temporal overlap
        for i in range(3):
            offset_x = base_x + np.random.uniform(-15, 15)
            offset_y = base_y + np.random.uniform(-15, 15)
            offset_z = base_z + np.random.uniform(-5, 5)
            
            waypoint_time = overlap_start + i * (overlap_end - overlap_start) / 2
            conflict_waypoints.append(Waypoint(offset_x, offset_y, offset_z, waypoint_time))
        
        return DroneMission(
            drone_id=conflict_id,
            waypoints=conflict_waypoints,
            start_time=overlap_start,
            end_time=overlap_end,
            safety_buffer=4.0
        )
    
    def _generate_near_miss(self, primary_mission: DroneMission, near_miss_id: str) -> DroneMission:
        """Generate a mission that creates a near-miss scenario."""
        # Find a point along primary trajectory
        primary_trajectory = primary_mission.get_trajectory_points(0.5)
        if not primary_trajectory:
            return self.generate_random_mission(near_miss_id)
        
        reference_point = primary_trajectory[len(primary_trajectory) // 2]
        ref_x, ref_y, ref_z, ref_time = reference_point
        
        # Create a mission that passes nearby but not directly conflicting
        near_miss_waypoints = []
        
        # Start point
        start_x = ref_x + np.random.uniform(-80, 80)
        start_y = ref_y + np.random.uniform(-80, 80)
        start_z = ref_z + np.random.uniform(-15, 15)
        start_time = ref_time - np.random.uniform(10, 20)
        
        near_miss_waypoints.append(Waypoint(start_x, start_y, start_z, start_time))
        
        # Near-miss point (within extended safety buffer but not immediate danger)
        miss_distance = np.random.uniform(8, 12)  # Just outside typical safety buffer
        angle = np.random.uniform(0, 2 * np.pi)
        
        miss_x = ref_x + miss_distance * np.cos(angle)
        miss_y = ref_y + miss_distance * np.sin(angle)
        miss_z = ref_z + np.random.uniform(-3, 3)
        miss_time = ref_time + np.random.uniform(-1, 1)
        
        near_miss_waypoints.append(Waypoint(miss_x, miss_y, miss_z, miss_time))
        
        # End point
        end_x = miss_x + np.random.uniform(-50, 50)
        end_y = miss_y + np.random.uniform(-50, 50)
        end_z = miss_z + np.random.uniform(-10, 10)
        end_time = miss_time + np.random.uniform(15, 25)
        
        near_miss_waypoints.append(Waypoint(end_x, end_y, end_z, end_time))
        
        return DroneMission(
            drone_id=near_miss_id,
            waypoints=near_miss_waypoints,
            start_time=start_time,
            end_time=end_time,
            safety_buffer=6.0
        )
    
    def generate_high_density_scenario(self, num_drones: int = 20, 
                                     airspace_size: Tuple[float, float] = (200, 200)) -> List[DroneMission]:
        """
        Generate a high-density airspace scenario for stress testing.
        
        Args:
            num_drones: Number of drones to generate
            airspace_size: Size of the airspace (width, height)
            
        Returns:
            List of drone missions in a constrained airspace
        """
        # Temporarily reduce airspace for high density
        original_bounds = self.airspace_bounds.copy()
        self.airspace_bounds.update({
            'x_max': airspace_size[0],
            'y_max': airspace_size[1]
        })
        
        try:
            missions = []
            
            # Generate overlapping time windows for higher conflict probability
            base_start_time = 0
            time_window = 120  # All missions within 2-minute window
            
            for i in range(num_drones):
                drone_id = f"DENSE_{i+1:03d}"
                start_time = base_start_time + np.random.uniform(0, time_window * 0.7)
                
                mission = self.generate_random_mission(
                    drone_id, 
                    start_time=start_time,
                    mission_duration=np.random.uniform(30, 90)
                )
                missions.append(mission)
            
            return missions
            
        finally:
            # Restore original bounds
            self.airspace_bounds = original_bounds
    
    def save_scenario_to_file(self, missions: List[DroneMission], filename: str):
        """
        Save a scenario to a JSON file for later use.
        
        Args:
            missions: List of missions to save
            filename: Output filename
        """
        import json
        
        scenario_data = {
            'metadata': {
                'generator_version': '1.0',
                'num_missions': len(missions),
                'airspace_bounds': self.airspace_bounds
            },
            'missions': []
        }
        
        for mission in missions:
            mission_data = {
                'drone_id': mission.drone_id,
                'start_time': mission.start_time,
                'end_time': mission.end_time,
                'safety_buffer': mission.safety_buffer,
                'waypoints': [
                    {
                        'x': wp.x,
                        'y': wp.y,
                        'z': wp.z,
                        'time': wp.time
                    } for wp in mission.waypoints
                ]
            }
            scenario_data['missions'].append(mission_data)
        
        with open(filename, 'w') as f:
            json.dump(scenario_data, f, indent=2)
    
    def load_scenario_from_file(self, filename: str) -> List[DroneMission]:
        """
        Load a scenario from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            List of loaded missions
        """
        import json
        
        with open(filename, 'r') as f:
            scenario_data = json.load(f)
        
        missions = []
        
        for mission_data in scenario_data['missions']:
            waypoints = []
            for wp_data in mission_data['waypoints']:
                waypoint = Waypoint(
                    wp_data['x'], wp_data['y'], 
                    wp_data['z'], wp_data['time']
                )
                waypoints.append(waypoint)
            
            mission = DroneMission(
                drone_id=mission_data['drone_id'],
                waypoints=waypoints,
                start_time=mission_data['start_time'],
                end_time=mission_data['end_time'],
                safety_buffer=mission_data['safety_buffer']
            )
            missions.append(mission)
        
        return missions