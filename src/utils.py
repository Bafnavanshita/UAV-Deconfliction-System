"""
Utilities Module

This module provides common utility functions for the UAV deconfliction system,
including logging setup, file I/O, and data processing helpers.
"""

import logging
import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import csv


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Set up logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger('uav_deconfliction')
    logger.info("Logging system initialized")
    
    return logger


def save_results(results: Dict[str, Any], filename: str):
    """
    Save analysis results to a JSON file.
    
    Args:
        results: Results dictionary to save
        filename: Output filename
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = convert_numpy_types(results)
    
    # Add timestamp
    json_results['saved_timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results saved to: {filename}")


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy types converted
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def load_results(filename: str) -> Dict[str, Any]:
    """
    Load analysis results from a JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded results dictionary
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results


def calculate_distance_3d(point1: Tuple[float, float, float], 
                         point2: Tuple[float, float, float]) -> float:
    """
    Calculate 3D Euclidean distance between two points.
    
    Args:
        point1: First point (x, y, z)
        point2: Second point (x, y, z)
        
    Returns:
        Distance in meters
    """
    return np.sqrt(
        (point1[0] - point2[0])**2 + 
        (point1[1] - point2[1])**2 + 
        (point1[2] - point2[2])**2
    )


def calculate_trajectory_length(trajectory_points: List[Tuple[float, float, float, float]]) -> float:
    """
    Calculate total length of a trajectory.
    
    Args:
        trajectory_points: List of (x, y, z, time) tuples
        
    Returns:
        Total trajectory length in meters
    """
    if len(trajectory_points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(trajectory_points)):
        prev_point = trajectory_points[i-1][:3]
        curr_point = trajectory_points[i][:3]
        total_length += calculate_distance_3d(prev_point, curr_point)
    
    return total_length


def find_closest_approach(trajectory1: List[Tuple[float, float, float, float]],
                         trajectory2: List[Tuple[float, float, float, float]]) -> Dict[str, Any]:
    """
    Find the point of closest approach between two trajectories.
    
    Args:
        trajectory1: First trajectory points
        trajectory2: Second trajectory points
        
    Returns:
        Dictionary with closest approach information
    """
    min_distance = float('inf')
    closest_info = None
    
    for point1 in trajectory1:
        for point2 in trajectory2:
            # Check if points are temporally close
            time_diff = abs(point1[3] - point2[3])
            if time_diff <= 2.0:  # Within 2 seconds
                distance = calculate_distance_3d(point1[:3], point2[:3])
                if distance < min_distance:
                    min_distance = distance
                    closest_info = {
                        'distance': distance,
                        'time1': point1[3],
                        'time2': point2[3],
                        'position1': point1[:3],
                        'position2': point2[:3],
                        'time_difference': time_diff
                    }
    
    return closest_info


def generate_conflict_report(conflicts: List[Dict[str, Any]], 
                           primary_mission_id: str) -> str:
    """
    Generate a human-readable conflict report.
    
    Args:
        conflicts: List of conflict dictionaries
        primary_mission_id: ID of the primary mission
        
    Returns:
        Formatted conflict report string
    """
    if not conflicts:
        return f"✅ MISSION APPROVED: {primary_mission_id}\nNo conflicts detected."
    
    report_lines = [
        f"⚠️  CONFLICT REPORT: {primary_mission_id}",
        f"Total Conflicts Detected: {len(conflicts)}",
        "=" * 50
    ]
    
    # Group conflicts by severity
    severity_groups = {}
    for conflict in conflicts:
        severity = conflict.get('severity', 'LOW')
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(conflict)
    
    # Report by severity (most critical first)
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING']
    
    for severity in severity_order:
        if severity in severity_groups:
            report_lines.append(f"\n{severity} CONFLICTS ({len(severity_groups[severity])}):")
            
            for i, conflict in enumerate(severity_groups[severity], 1):
                report_lines.extend([
                    f"  {i}. Time: {conflict['time']:.1f}s",
                    f"     Location: ({conflict['location'][0]:.1f}, {conflict['location'][1]:.1f}, {conflict['location'][2]:.1f})",
                    f"     Distance: {conflict['distance']:.1f}m",
                    f"     Conflicting Drone: {conflict['conflicting_drone']}",
                    f"     Description: {conflict['description']}"
                ])
    
    # Add recommendations
    report_lines.extend([
        "\n" + "=" * 50,
        "RECOMMENDATIONS:",
    ])
    
    if any(c.get('severity') in ['CRITICAL', 'HIGH'] for c in conflicts):
        report_lines.append("🚫 MISSION REJECTED - Immediate replanning required")
    elif any(c.get('severity') == 'MEDIUM' for c in conflicts):
        report_lines.append("⚠️  MISSION REQUIRES REVIEW - Consider alternative timing")
    else:
        report_lines.append("⚡ MISSION APPROVED WITH CAUTION - Monitor during execution")
    
    return "\n".join(report_lines)


def export_mission_data_csv(missions: List, filename: str, include_trajectories: bool = False):
    """
    Export mission data to CSV format.
    
    Args:
        missions: List of DroneMission objects
        filename: Output CSV filename
        include_trajectories: Whether to include detailed trajectory points
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as csvfile:
        if include_trajectories:
            fieldnames = ['drone_id', 'time', 'x', 'y', 'z', 'waypoint_index', 'mission_start', 'mission_end']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for mission in missions:
                trajectory = mission.get_trajectory_points(1.0)
                waypoint_index = 0
                
                for point in trajectory:
                    # Check if this point is a waypoint
                    is_waypoint = False
                    for i, wp in enumerate(mission.waypoints):
                        if abs(wp.time - point[3]) < 0.1:
                            waypoint_index = i
                            is_waypoint = True
                            break
                    
                    writer.writerow({
                        'drone_id': mission.drone_id,
                        'time': point[3],
                        'x': point[0],
                        'y': point[1],
                        'z': point[2],
                        'waypoint_index': waypoint_index if is_waypoint else '',
                        'mission_start': mission.start_time,
                        'mission_end': mission.end_time
                    })
        else:
            fieldnames = ['drone_id', 'start_time', 'end_time', 'num_waypoints', 'total_distance', 'safety_buffer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for mission in missions:
                summary = mission.get_mission_summary()
                writer.writerow({
                    'drone_id': mission.drone_id,
                    'start_time': mission.start_time,
                    'end_time': mission.end_time,
                    'num_waypoints': len(mission.waypoints),
                    'total_distance': summary['total_distance'],
                    'safety_buffer': mission.safety_buffer
                })


def validate_airspace_bounds(waypoints: List[Tuple[float, float, float]], 
                           bounds: Dict[str, float]) -> List[str]:
    """
    Validate that waypoints are within airspace boundaries.
    
    Args:
        waypoints: List of (x, y, z) waypoint coordinates
        bounds: Dictionary with airspace boundaries
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    for i, (x, y, z) in enumerate(waypoints):
        if x < bounds.get('x_min', 0) or x > bounds.get('x_max', 1000):
            errors.append(f"Waypoint {i+1}: X coordinate {x} outside bounds [{bounds.get('x_min', 0)}, {bounds.get('x_max', 1000)}]")
        
        if y < bounds.get('y_min', 0) or y > bounds.get('y_max', 1000):
            errors.append(f"Waypoint {i+1}: Y coordinate {y} outside bounds [{bounds.get('y_min', 0)}, {bounds.get('y_max', 1000)}]")
        
        if z < bounds.get('z_min', 0) or z > bounds.get('z_max', 500):
            errors.append(f"Waypoint {i+1}: Z coordinate {z} outside bounds [{bounds.get('z_min', 0)}, {bounds.get('z_max', 500)}]")
    
    return errors


def create_performance_metrics(conflicts: List[Dict[str, Any]], 
                             processing_time: float,
                             num_missions_checked: int) -> Dict[str, Any]:
    """
    Create performance metrics for the deconfliction system.
    
    Args:
        conflicts: List of detected conflicts
        processing_time: Time taken for processing in seconds
        num_missions_checked: Number of missions processed
        
    Returns:
        Dictionary with performance metrics
    """
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'WARNING': 0}
    
    for conflict in conflicts:
        severity = conflict.get('severity', 'LOW')
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    return {
        'total_conflicts': len(conflicts),
        'severity_breakdown': severity_counts,
        'processing_time_seconds': processing_time,
        'missions_checked': num_missions_checked,
        'conflicts_per_mission': len(conflicts) / max(num_missions_checked, 1),
        'processing_rate_missions_per_second': num_missions_checked / max(processing_time, 0.001),
        'critical_conflict_rate': severity_counts['CRITICAL'] / max(len(conflicts), 1)
    }


def format_time_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def calculate_airspace_utilization(missions: List, time_window: Tuple[float, float], 
                                 time_resolution: float = 1.0) -> Dict[str, Any]:
    """
    Calculate airspace utilization statistics over time.
    
    Args:
        missions: List of DroneMission objects
        time_window: (start_time, end_time) tuple
        time_resolution: Time step for analysis
        
    Returns:
        Dictionary with utilization statistics
    """
    start_time, end_time = time_window
    time_points = np.arange(start_time, end_time + time_resolution, time_resolution)
    
    utilization_data = []
    active_counts = []
    
    for t in time_points:
        active_drones = 0
        active_drone_ids = []
        
        for mission in missions:
            if mission.is_active_at_time(t):
                active_drones += 1
                active_drone_ids.append(mission.drone_id)
        
        utilization_data.append({
            'time': t,
            'active_count': active_drones,
            'active_drones': active_drone_ids
        })
        active_counts.append(active_drones)
    
    return {
        'time_window': time_window,
        'peak_utilization': max(active_counts) if active_counts else 0,
        'average_utilization': np.mean(active_counts) if active_counts else 0,
        'minimum_utilization': min(active_counts) if active_counts else 0,
        'utilization_timeline': utilization_data,
        'total_missions': len(missions),
        'time_resolution': time_resolution
    }


class ColorManager:
    """Utility class for managing consistent color schemes across visualizations."""
    
    def __init__(self):
        self.color_schemes = {
            'severity': {
                'CRITICAL': '#FF0000',
                'HIGH': '#FF8800',
                'MEDIUM': '#FFFF00',
                'LOW': '#FFB6C1',
                'WARNING': '#0000FF'
            },
            'drone_types': {
                'primary': '#FF0000',
                'delivery': '#00AA00',
                'surveillance': '#0066CC',
                'inspection': '#AA00AA',
                'emergency': '#FF6600'
            },
            'status': {
                'clear': '#00AA00',
                'warning': '#FFAA00',
                'conflict': '#FF0000'
            }
        }
    
    def get_color(self, category: str, key: str) -> str:
        """Get color for a specific category and key."""
        return self.color_schemes.get(category, {}).get(key, '#808080')
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """Generate a diverse color palette."""
        import colorsys
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
        return colors


def create_summary_statistics(missions: List, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics for a deconfliction analysis.
    
    Args:
        missions: List of DroneMission objects
        conflicts: List of detected conflicts
        
    Returns:
        Dictionary with summary statistics
    """
    if not missions:
        return {'error': 'No missions provided'}
    
    # Mission statistics
    mission_durations = [m.end_time - m.start_time for m in missions]
    mission_starts = [m.start_time for m in missions]
    mission_ends = [m.end_time for m in missions]
    
    # Conflict statistics
    conflict_times = [c['time'] for c in conflicts] if conflicts else []
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'WARNING': 0}
    
    for conflict in conflicts:
        severity = conflict.get('severity', 'LOW')
        if severity in severity_counts:
            severity_counts[severity] += 1
    
    # Calculate conflict density (conflicts per unit time)
    total_mission_time = sum(mission_durations)
    conflict_density = len(conflicts) / max(total_mission_time, 1)
    
    return {
        'mission_summary': {
            'total_missions': len(missions),
            'average_duration': np.mean(mission_durations),
            'total_flight_time': total_mission_time,
            'earliest_start': min(mission_starts),
            'latest_end': max(mission_ends),
            'peak_concurrent_missions': calculate_peak_concurrency(missions)
        },
        'conflict_summary': {
            'total_conflicts': len(conflicts),
            'severity_breakdown': severity_counts,
            'conflict_density_per_hour': conflict_density * 3600,
            'first_conflict_time': min(conflict_times) if conflict_times else None,
            'last_conflict_time': max(conflict_times) if conflict_times else None
        },
        'safety_metrics': {
            'conflict_free_missions': len([m for m in missions if not any(c['primary_drone'] == m.drone_id or c['conflicting_drone'] == m.drone_id for c in conflicts)]),
            'high_risk_missions': len([m for m in missions if any((c['primary_drone'] == m.drone_id or c['conflicting_drone'] == m.drone_id) and c.get('severity') in ['CRITICAL', 'HIGH'] for c in conflicts)]),
            'overall_safety_score': calculate_safety_score(missions, conflicts)
        }
    }


def calculate_peak_concurrency(missions: List) -> int:
    """Calculate the maximum number of concurrent missions."""
    if not missions:
        return 0
    
    # Get all time events (starts and ends)
    events = []
    for mission in missions:
        events.append((mission.start_time, 'start'))
        events.append((mission.end_time, 'end'))
    
    # Sort events by time
    events.sort()
    
    current_count = 0
    max_count = 0
    
    for time, event_type in events:
        if event_type == 'start':
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count -= 1
    
    return max_count


def calculate_safety_score(missions: List, conflicts: List[Dict[str, Any]]) -> float:
    """
    Calculate an overall safety score (0-100) based on missions and conflicts.
    
    Args:
        missions: List of missions
        conflicts: List of conflicts
        
    Returns:
        Safety score from 0 (very unsafe) to 100 (completely safe)
    """
    if not missions:
        return 100.0
    
    base_score = 100.0
    
    # Deduct points based on conflict severity
    severity_penalties = {'CRITICAL': 25, 'HIGH': 15, 'MEDIUM': 8, 'LOW': 3, 'WARNING': 1}
    
    for conflict in conflicts:
        severity = conflict.get('severity', 'LOW')
        penalty = severity_penalties.get(severity, 1)
        base_score -= penalty
    
    # Additional penalty for high conflict density
    conflict_rate = len(conflicts) / len(missions)
    if conflict_rate > 0.5:  # More than 0.5 conflicts per mission
        base_score -= 10
    elif conflict_rate > 0.2:
        base_score -= 5
    
    return max(0.0, min(100.0, base_score))