"""
Drone Mission Module

This module defines the core data structures for representing drone missions,
waypoints, and trajectories in the UAV deconfliction system.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d
import math

@dataclass
class Waypoint:
    """Represents a single waypoint in 3D space with timing information."""
    x: float
    y: float
    z: float
    time: float
    
    def distance_to(self, other: 'Waypoint') -> float:
        """Calculate Euclidean distance to another waypoint."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def __str__(self) -> str:
        return f"Waypoint({self.x:.1f}, {self.y:.1f}, {self.z:.1f}, t={self.time:.1f})"

class DroneMission:
    """
    Represents a complete drone mission with waypoints and timing constraints.
    
    Provides methods for trajectory interpolation and conflict detection support.
    """
    
    def __init__(self, drone_id: str, waypoints: List[Waypoint], 
                 start_time: float, end_time: float, safety_buffer: float = 5.0):
        self.drone_id = drone_id
        self.waypoints = waypoints
        self.start_time = start_time
        self.end_time = end_time
        self.safety_buffer = safety_buffer
        
        # Validate mission parameters
        self._validate_mission()
        
        # Create interpolated trajectory
        self._create_trajectory()
    
    def _validate_mission(self):
        """Validate mission parameters and waypoint consistency."""
        if not self.waypoints:
            raise ValueError("Mission must have at least one waypoint")
        
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        
        if self.safety_buffer <= 0:
            raise ValueError("Safety buffer must be positive")
        
        # Check waypoint timing consistency
        for waypoint in self.waypoints:
            if not (self.start_time <= waypoint.time <= self.end_time):
                raise ValueError(f"Waypoint time {waypoint.time} outside mission window [{self.start_time}, {self.end_time}]")
    
    def _create_trajectory(self):
        """Create interpolated trajectory functions for smooth path planning."""
        if len(self.waypoints) < 2:
            # Single waypoint - stationary mission
            self.x_interp = lambda t: self.waypoints[0].x
            self.y_interp = lambda t: self.waypoints[0].y
            self.z_interp = lambda t: self.waypoints[0].z
            return
        
        # Extract coordinates and times
        times = [wp.time for wp in self.waypoints]
        x_coords = [wp.x for wp in self.waypoints]
        y_coords = [wp.y for wp in self.waypoints]
        z_coords = [wp.z for wp in self.waypoints]
        
        # Create interpolation functions
        # Use linear interpolation for smooth, predictable trajectories
        self.x_interp = interp1d(times, x_coords, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
        self.y_interp = interp1d(times, y_coords, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
        self.z_interp = interp1d(times, z_coords, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
    
    def get_position_at_time(self, time: float) -> Tuple[float, float, float]:
        """
        Get the interpolated position of the drone at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        if time < self.start_time or time > self.end_time:
            # Drone is not active at this time
            return None
        
        x = float(self.x_interp(time))
        y = float(self.y_interp(time))
        z = float(self.z_interp(time))
        
        return (x, y, z)
    
    def get_trajectory_points(self, time_resolution: float = 1.0) -> List[Tuple[float, float, float, float]]:
        """
        Get discretized trajectory points for visualization and analysis.
        
        Args:
            time_resolution: Time step between points in seconds
            
        Returns:
            List of (x, y, z, time) tuples
        """
        points = []
        current_time = self.start_time
        
        while current_time <= self.end_time:
            pos = self.get_position_at_time(current_time)
            if pos is not None:
                points.append((pos[0], pos[1], pos[2], current_time))
            current_time += time_resolution
        
        return points
    
    def get_bounding_box(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get the 3D bounding box of the mission trajectory.
        
        Returns:
            Tuple of ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        """
        trajectory_points = self.get_trajectory_points(0.5)  # High resolution for accurate bounds
        
        if not trajectory_points:
            # Single waypoint case
            wp = self.waypoints[0]
            return ((wp.x, wp.x), (wp.y, wp.y), (wp.z, wp.z))
        
        x_coords = [point[0] for point in trajectory_points]
        y_coords = [point[1] for point in trajectory_points]
        z_coords = [point[2] for point in trajectory_points]
        
        return (
            (min(x_coords), max(x_coords)),
            (min(y_coords), max(y_coords)),
            (min(z_coords), max(z_coords))
        )
    
    def is_active_at_time(self, time: float) -> bool:
        """Check if the drone is active (flying) at a given time."""
        return self.start_time <= time <= self.end_time
    
    def get_speed_profile(self) -> List[Tuple[float, float]]:
        """
        Calculate speed profile of the mission.
        
        Returns:
            List of (time, speed) tuples
        """
        if len(self.waypoints) < 2:
            return [(self.start_time, 0.0), (self.end_time, 0.0)]
        
        speed_profile = []
        
        for i in range(len(self.waypoints) - 1):
            wp1 = self.waypoints[i]
            wp2 = self.waypoints[i + 1]
            
            distance = wp1.distance_to(wp2)
            time_diff = wp2.time - wp1.time
            
            if time_diff > 0:
                speed = distance / time_diff
            else:
                speed = 0.0
            
            speed_profile.append((wp1.time, speed))
        
        # Add final waypoint with zero speed
        speed_profile.append((self.waypoints[-1].time, 0.0))
        
        return speed_profile
    
    def get_mission_summary(self) -> dict:
        """Get a summary of mission parameters and statistics."""
        trajectory_points = self.get_trajectory_points(1.0)
        total_distance = 0.0
        
        # Calculate total distance
        for i in range(len(self.waypoints) - 1):
            total_distance += self.waypoints[i].distance_to(self.waypoints[i + 1])
        
        # Calculate average speed
        mission_duration = self.end_time - self.start_time
        avg_speed = total_distance / mission_duration if mission_duration > 0 else 0.0
        
        bounds = self.get_bounding_box()
        
        return {
            'drone_id': self.drone_id,
            'waypoint_count': len(self.waypoints),
            'mission_duration': mission_duration,
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'safety_buffer': self.safety_buffer,
            'bounding_box': bounds,
            'start_time': self.start_time,
            'end_time': self.end_time
        }
    
    def __str__(self) -> str:
        return f"DroneMission({self.drone_id}, {len(self.waypoints)} waypoints, {self.start_time}-{self.end_time}s)"
    
    def __repr__(self) -> str:
        return self.__str__()