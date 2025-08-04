"""
Deconfliction Service Module

This module implements the core conflict detection logic for the UAV
deconfliction system, including spatial and temporal conflict analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
import logging

from .drone_mission import DroneMission

class ConflictDetector:
    """Handles the core conflict detection algorithms."""
    
    def __init__(self, time_resolution: float = 0.5):
        """
        Initialize conflict detector.
        
        Args:
            time_resolution: Time step for trajectory sampling in seconds
        """
        self.time_resolution = time_resolution
        self.logger = logging.getLogger(__name__)
    
    def detect_spatial_temporal_conflicts(self, primary: DroneMission, 
                                        other: DroneMission) -> List[Dict[str, Any]]:
        """
        Detect all conflicts between two drone missions.
        
        Args:
            primary: Primary drone mission
            other: Other drone mission to check against
            
        Returns:
            List of conflict dictionaries
        """
        conflicts = []
        
        # Find overlapping time window
        overlap_start = max(primary.start_time, other.start_time)
        overlap_end = min(primary.end_time, other.end_time)
        
        if overlap_start >= overlap_end:
            # No temporal overlap
            return conflicts
        
        # Sample trajectories at regular intervals
        current_time = overlap_start
        min_distance = float('inf')
        closest_encounter = None
        
        while current_time <= overlap_end:
            primary_pos = primary.get_position_at_time(current_time)
            other_pos = other.get_position_at_time(current_time)
            
            if primary_pos is not None and other_pos is not None:
                # Calculate 3D distance
                distance = math.sqrt(
                    (primary_pos[0] - other_pos[0])**2 +
                    (primary_pos[1] - other_pos[1])**2 +
                    (primary_pos[2] - other_pos[2])**2
                )
                
                # Track closest encounter
                if distance < min_distance:
                    min_distance = distance
                    closest_encounter = {
                        'time': current_time,
                        'primary_pos': primary_pos,
                        'other_pos': other_pos,
                        'distance': distance
                    }
                
                # Check if within safety buffer
                combined_buffer = primary.safety_buffer + other.safety_buffer
                if distance < combined_buffer:
                    conflict = self._create_conflict_record(
                        primary, other, current_time, 
                        primary_pos, other_pos, distance, combined_buffer
                    )
                    conflicts.append(conflict)
            
            current_time += self.time_resolution
        
        # If no direct conflicts but close encounter, create warning
        if not conflicts and closest_encounter and min_distance < combined_buffer * 1.5:
            warning = self._create_warning_record(
                primary, other, closest_encounter, combined_buffer
            )
            conflicts.append(warning)
        
        return conflicts
    
    def _create_conflict_record(self, primary: DroneMission, other: DroneMission,
                              time: float, pos1: Tuple[float, float, float],
                              pos2: Tuple[float, float, float], distance: float,
                              required_separation: float) -> Dict[str, Any]:
        """Create a detailed conflict record."""
        
        # Determine conflict severity
        severity_ratio = distance / required_separation
        if severity_ratio < 0.3:
            severity = "CRITICAL"
        elif severity_ratio < 0.6:
            severity = "HIGH"
        elif severity_ratio < 0.8:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Calculate midpoint for conflict location
        conflict_location = (
            (pos1[0] + pos2[0]) / 2,
            (pos1[1] + pos2[1]) / 2,
            (pos1[2] + pos2[2]) / 2
        )
        
        return {
            'type': 'spatial_temporal_conflict',
            'severity': severity,
            'time': time,
            'location': conflict_location,
            'distance': distance,
            'required_separation': required_separation,
            'primary_drone': primary.drone_id,
            'conflicting_drone': other.drone_id,
            'primary_position': pos1,
            'conflicting_position': pos2,
            'description': f"Conflict between {primary.drone_id} and {other.drone_id} at t={time:.1f}s: {distance:.1f}m separation (required: {required_separation:.1f}m)"
        }
    
    def _create_warning_record(self, primary: DroneMission, other: DroneMission,
                             encounter: Dict[str, Any], required_separation: float) -> Dict[str, Any]:
        """Create a warning record for close encounters."""
        
        conflict_location = (
            (encounter['primary_pos'][0] + encounter['other_pos'][0]) / 2,
            (encounter['primary_pos'][1] + encounter['other_pos'][1]) / 2,
            (encounter['primary_pos'][2] + encounter['other_pos'][2]) / 2
        )
        
        return {
            'type': 'close_encounter_warning',
            'severity': 'WARNING',
            'time': encounter['time'],
            'location': conflict_location,
            'distance': encounter['distance'],
            'required_separation': required_separation,
            'primary_drone': primary.drone_id,
            'conflicting_drone': other.drone_id,
            'primary_position': encounter['primary_pos'],
            'conflicting_position': encounter['other_pos'],
            'description': f"Close encounter between {primary.drone_id} and {other.drone_id} at t={encounter['time']:.1f}s: {encounter['distance']:.1f}m separation"
        }

class DeconflictionService:
    """
    Main deconfliction service that manages multiple drone missions
    and provides conflict detection capabilities.
    """
    
    def __init__(self, time_resolution: float = 0.5, max_workers: int = 4):
        """
        Initialize the deconfliction service.
        
        Args:
            time_resolution: Time step for conflict detection in seconds
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.simulated_flights: List[DroneMission] = []
        self.conflict_detector = ConflictDetector(time_resolution)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.total_checks = 0
        self.total_conflicts = 0
    
    def add_simulated_flight(self, mission: DroneMission):
        """Add a simulated flight to the deconfliction database."""
        if not isinstance(mission, DroneMission):
            raise TypeError("Mission must be a DroneMission instance")
        
        self.simulated_flights.append(mission)
        self.logger.info(f"Added simulated flight: {mission.drone_id}")
    
    def remove_simulated_flight(self, drone_id: str) -> bool:
        """Remove a simulated flight by drone ID."""
        for i, mission in enumerate(self.simulated_flights):
            if mission.drone_id == drone_id:
                del self.simulated_flights[i]
                self.logger.info(f"Removed simulated flight: {drone_id}")
                return True
        return False
    
    def clear_simulated_flights(self):
        """Clear all simulated flights."""
        self.simulated_flights.clear()
        self.logger.info("Cleared all simulated flights")
    
    def check_mission_conflicts(self, primary_mission: DroneMission) -> Dict[str, Any]:
        """
        Check a primary mission against all simulated flights for conflicts.
        
        Args:
            primary_mission: The mission to validate
            
        Returns:
            Dictionary containing conflict analysis results
        """
        self.logger.info(f"Checking conflicts for mission: {primary_mission.drone_id}")
        
        start_time = primary_mission.start_time
        end_time = primary_mission.end_time
        
        all_conflicts = []
        conflict_summary = {
            'total_conflicts': 0,
            'critical_conflicts': 0,
            'high_conflicts': 0,
            'medium_conflicts': 0,
            'low_conflicts': 0,
            'warnings': 0
        }
        
        # Check against each simulated flight
        if self.max_workers > 1 and len(self.simulated_flights) > 2:
            # Use parallel processing for multiple flights
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for simulated_flight in self.simulated_flights:
                    future = executor.submit(
                        self.conflict_detector.detect_spatial_temporal_conflicts,
                        primary_mission, simulated_flight
                    )
                    futures.append(future)
                
                for future in futures:
                    conflicts = future.result()
                    all_conflicts.extend(conflicts)
        else:
            # Sequential processing
            for simulated_flight in self.simulated_flights:
                conflicts = self.conflict_detector.detect_spatial_temporal_conflicts(
                    primary_mission, simulated_flight
                )
                all_conflicts.extend(conflicts)
        
        # Analyze conflicts
        for conflict in all_conflicts:
            conflict_summary['total_conflicts'] += 1
            
            severity = conflict.get('severity', 'LOW')
            if severity == 'CRITICAL':
                conflict_summary['critical_conflicts'] += 1
            elif severity == 'HIGH':
                conflict_summary['high_conflicts'] += 1
            elif severity == 'MEDIUM':
                conflict_summary['medium_conflicts'] += 1
            elif severity == 'WARNING':
                conflict_summary['warnings'] += 1
            else:
                conflict_summary['low_conflicts'] += 1
        
        # Sort conflicts by time
        all_conflicts.sort(key=lambda x: x['time'])
        
        # Determine overall status
        if conflict_summary['critical_conflicts'] > 0 or conflict_summary['high_conflicts'] > 0:
            status = 'conflict_detected'
        elif conflict_summary['medium_conflicts'] > 0:
            status = 'conflict_detected'
        elif conflict_summary['warnings'] > 0:
            status = 'warning'
        else:
            status = 'clear'
        
        # Update performance tracking
        self.total_checks += 1
        self.total_conflicts += conflict_summary['total_conflicts']
        
        result = {
            'status': status,
            'primary_mission': primary_mission.drone_id,
            'check_timestamp': np.datetime64('now'),
            'conflicts': all_conflicts,
            'conflict_summary': conflict_summary,
            'simulated_flights_checked': len(self.simulated_flights),
            'recommendation': self._generate_recommendation(status, conflict_summary),
            'alternative_time_windows': self._suggest_alternative_windows(primary_mission, all_conflicts)
        }
        
        self.logger.info(f"Conflict check complete: {status} ({len(all_conflicts)} conflicts)")
        return result
    
    def _generate_recommendation(self, status: str, summary: Dict[str, int]) -> str:
        """Generate mission recommendation based on conflict analysis."""
        if status == 'clear':
            return "Mission approved for execution. No conflicts detected."
        elif status == 'warning':
            return "Mission approved with caution. Monitor close encounters during execution."
        else:
            if summary['critical_conflicts'] > 0:
                return "Mission REJECTED. Critical conflicts detected. Immediate replanning required."
            elif summary['high_conflicts'] > 0:
                return "Mission REJECTED. High-risk conflicts detected. Replanning recommended."
            else:
                return "Mission requires review. Medium-risk conflicts detected."
    
    def _suggest_alternative_windows(self, primary_mission: DroneMission, 
                                   conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest alternative time windows to avoid conflicts."""
        if not conflicts:
            return []
        
        alternatives = []
        mission_duration = primary_mission.end_time - primary_mission.start_time
        
        # Analyze conflict time periods
        conflict_times = [c['time'] for c in conflicts]
        if conflict_times:
            earliest_conflict = min(conflict_times)
            latest_conflict = max(conflict_times)
            
            # Suggest earlier time window
            if earliest_conflict > mission_duration:
                alt_start = max(0, earliest_conflict - mission_duration - 10)
                alt_end = alt_start + mission_duration
                alternatives.append({
                    'type': 'earlier_window',
                    'start_time': alt_start,
                    'end_time': alt_end,
                    'description': f'Execute mission earlier to avoid conflicts'
                })
            
            # Suggest later time window
            alt_start = latest_conflict + 10
            alt_end = alt_start + mission_duration
            alternatives.append({
                'type': 'later_window',
                'start_time': alt_start,
                'end_time': alt_end,
                'description': f'Execute mission later to avoid conflicts'
            })
        
        return alternatives
    
    def get_airspace_status(self, time_window: Tuple[float, float] = None) -> Dict[str, Any]:
        """Get overall airspace status and utilization."""
        if time_window is None:
            # Analyze entire time range of all missions
            all_times = []
            for mission in self.simulated_flights:
                all_times.extend([mission.start_time, mission.end_time])
            
            if all_times:
                time_window = (min(all_times), max(all_times))
            else:
                time_window = (0, 100)  # Default window
        
        start_time, end_time = time_window
        
        # Calculate airspace utilization over time
        time_step = 1.0
        utilization_data = []
        
        for t in np.arange(start_time, end_time + time_step, time_step):
            active_drones = 0
            for mission in self.simulated_flights:
                if mission.is_active_at_time(t):
                    active_drones += 1
            
            utilization_data.append({
                'time': t,
                'active_drones': active_drones
            })
        
        # Calculate statistics
        active_counts = [data['active_drones'] for data in utilization_data]
        
        return {
            'time_window': time_window,
            'total_registered_flights': len(self.simulated_flights),
            'peak_utilization': max(active_counts) if active_counts else 0,
            'average_utilization': np.mean(active_counts) if active_counts else 0,
            'utilization_timeline': utilization_data,
            'total_checks_performed': self.total_checks,
            'total_conflicts_detected': self.total_conflicts
        }
    
    def batch_conflict_check(self, missions: List[DroneMission]) -> Dict[str, Any]:
        """
        Perform conflict checking for multiple missions simultaneously.
        
        Args:
            missions: List of missions to check
            
        Returns:
            Batch analysis results
        """
        results = {}
        
        for mission in missions:
            result = self.check_mission_conflicts(mission)
            results[mission.drone_id] = result
        
        # Generate batch summary
        total_conflicts = sum(len(result['conflicts']) for result in results.values())
        approved_missions = sum(1 for result in results.values() if result['status'] == 'clear')
        
        batch_summary = {
            'total_missions_checked': len(missions),
            'approved_missions': approved_missions,
            'rejected_missions': len(missions) - approved_missions,
            'total_conflicts_found': total_conflicts,
            'individual_results': results
        }
        
        return batch_summary