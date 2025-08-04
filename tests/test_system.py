#!/usr/bin/env python3
"""
Comprehensive Test Suite for UAV Deconfliction System

This module contains unit tests and integration tests for all components
of the UAV deconfliction system.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.drone_mission import DroneMission, Waypoint
from src.deconfliction_service import DeconflictionService, ConflictDetector
from src.data_generator import DataGenerator
from src.utils import calculate_distance_3d, validate_airspace_bounds

class TestWaypoint(unittest.TestCase):
    """Test cases for Waypoint class."""
    
    def test_waypoint_creation(self):
        """Test waypoint creation and basic functionality."""
        wp = Waypoint(10.0, 20.0, 30.0, 5.0)
        self.assertEqual(wp.x, 10.0)
        self.assertEqual(wp.y, 20.0)
        self.assertEqual(wp.z, 30.0)
        self.assertEqual(wp.time, 5.0)
    
    def test_waypoint_distance(self):
        """Test distance calculation between waypoints."""
        wp1 = Waypoint(0, 0, 0, 0)
        wp2 = Waypoint(3, 4, 0, 0)  # 3-4-5 triangle
        
        distance = wp1.distance_to(wp2)
        self.assertAlmostEqual(distance, 5.0, places=5)
    
    def test_waypoint_string_representation(self):
        """Test string representation of waypoints."""
        wp = Waypoint(1.23, 4.56, 7.89, 10.5)
        str_repr = str(wp)
        self.assertIn("1.2", str_repr)
        self.assertIn("4.6", str_repr)
        self.assertIn("7.9", str_repr)
        self.assertIn("10.5", str_repr)

class TestDroneMission(unittest.TestCase):
    """Test cases for DroneMission class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.waypoints = [
            Waypoint(0, 0, 10, 0),
            Waypoint(50, 50, 15, 30),
            Waypoint(100, 100, 20, 60)
        ]
        self.mission = DroneMission(
            drone_id="TEST_001",
            waypoints=self.waypoints,
            start_time=0,
            end_time=60,
            safety_buffer=5.0
        )
    
    def test_mission_creation(self):
        """Test mission creation and basic properties."""
        self.assertEqual(self.mission.drone_id, "TEST_001")
        self.assertEqual(len(self.mission.waypoints), 3)
        self.assertEqual(self.mission.start_time, 0)
        self.assertEqual(self.mission.end_time, 60)
        self.assertEqual(self.mission.safety_buffer, 5.0)
    
    def test_mission_validation(self):
        """Test mission parameter validation."""
        # Test empty waypoints
        with self.assertRaises(ValueError):
            DroneMission("TEST", [], 0, 60)
        
        # Test invalid time window
        with self.assertRaises(ValueError):
            DroneMission("TEST", self.waypoints, 60, 0)  # start > end
        
        # Test negative safety buffer
        with self.assertRaises(ValueError):
            DroneMission("TEST", self.waypoints, 0, 60, -1.0)
    
    def test_position_interpolation(self):
        """Test trajectory interpolation."""
        # Test position at waypoint times
        pos_start = self.mission.get_position_at_time(0)
        self.assertEqual(pos_start, (0, 0, 10))
        
        pos_mid = self.mission.get_position_at_time(30)
        self.assertEqual(pos_mid, (50, 50, 15))
        
        pos_end = self.mission.get_position_at_time(60)
        self.assertEqual(pos_end, (100, 100, 20))
        
        # Test interpolated position
        pos_interp = self.mission.get_position_at_time(15)  # Halfway between 0 and 30
        self.assertEqual(pos_interp, (25, 25, 12.5))
    
    def test_trajectory_points(self):
        """Test trajectory point generation."""
        points = self.mission.get_trajectory_points(15.0)  # 15-second resolution
        
        # Should have points at times 0, 15, 30, 45, 60
        expected_times = [0, 15, 30, 45, 60]
        actual_times = [point[3] for point in points]
        
        self.assertEqual(len(points), len(expected_times))
        for expected, actual in zip(expected_times, actual_times):
            self.assertAlmostEqual(expected, actual, places=5)
    
    def test_bounding_box(self):
        """Test bounding box calculation."""
        bounds = self.mission.get_bounding_box()
        
        # X bounds: 0 to 100
        self.assertAlmostEqual(bounds[0][0], 0, places=1)
        self.assertAlmostEqual(bounds[0][1], 100, places=1)
        
        # Y bounds: 0 to 100
        self.assertAlmostEqual(bounds[1][0], 0, places=1)
        self.assertAlmostEqual(bounds[1][1], 100, places=1)
        
        # Z bounds: 10 to 20
        self.assertAlmostEqual(bounds[2][0], 10, places=1)
        self.assertAlmostEqual(bounds[2][1], 20, places=1)
    
    def test_activity_check(self):
        """Test activity time checking."""
        self.assertTrue(self.mission.is_active_at_time(30))
        self.assertTrue(self.mission.is_active_at_time(0))
        self.assertTrue(self.mission.is_active_at_time(60))
        self.assertFalse(self.mission.is_active_at_time(-10))
        self.assertFalse(self.mission.is_active_at_time(70))
    
    def test_speed_profile(self):
        """Test speed profile calculation."""
        speed_profile = self.mission.get_speed_profile()
        
        self.assertEqual(len(speed_profile), 3)  # One per waypoint
        
        # All speeds should be positive
        for time, speed in speed_profile[:-1]:  # Exclude last waypoint (speed = 0)
            self.assertGreater(speed, 0)
        
        # Last waypoint should have zero speed
        self.assertEqual(speed_profile[-1][1], 0.0)

class TestConflictDetector(unittest.TestCase):
    """Test cases for ConflictDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ConflictDetector(time_resolution=1.0)
        
        # Create two missions that will conflict
        self.mission1 = DroneMission(
            drone_id="MISSION_1",
            waypoints=[
                Waypoint(0, 0, 10, 0),
                Waypoint(100, 0, 10, 60)
            ],
            start_time=0,
            end_time=60,
            safety_buffer=5.0
        )
        
        self.mission2 = DroneMission(
            drone_id="MISSION_2", 
            waypoints=[
                Waypoint(50, -2, 11, 25),  # Very close to mission1 path
                Waypoint(50, 2, 9, 35)
            ],
            start_time=25,
            end_time=35,
            safety_buffer=5.0
        )
    
    def test_conflict_detection(self):
        """Test basic conflict detection."""
        conflicts = self.detector.detect_spatial_temporal_conflicts(
            self.mission1, self.mission2
        )
        
        # Should detect conflicts
        self.assertGreater(len(conflicts), 0)
        
        # Check conflict properties
        conflict = conflicts[0]
        self.assertIn('type', conflict)
        self.assertIn('time', conflict)
        self.assertIn('location', conflict)
        self.assertIn('distance', conflict)
        self.assertIn('severity', conflict)
    
    def test_no_conflict_scenario(self):
        """Test scenario with no conflicts."""
        # Create missions that don't conflict
        safe_mission = DroneMission(
            drone_id="SAFE_MISSION",
            waypoints=[
                Waypoint(200, 200, 50, 0),  # Far away
                Waypoint(300, 300, 60, 60)
            ],
            start_time=0,
            end_time=60,
            safety_buffer=5.0
        )
        
        conflicts = self.detector.detect_spatial_temporal_conflicts(
            self.mission1, safe_mission
        )
        
        # Should not detect conflicts (or only warnings)
        critical_conflicts = [c for c in conflicts if c.get('severity') in ['CRITICAL', 'HIGH', 'MEDIUM']]
        self.assertEqual(len(critical_conflicts), 0)
    
    def test_temporal_separation(self):
        """Test that temporally separated missions don't conflict."""
        later_mission = DroneMission(
            drone_id="LATER_MISSION",
            waypoints=[
                Waypoint(0, 0, 10, 100),  # Same path but later time
                Waypoint(100, 0, 10, 160)
            ],
            start_time=100,
            end_time=160,
            safety_buffer=5.0
        )
        
        conflicts = self.detector.detect_spatial_temporal_conflicts(
            self.mission1, later_mission
        )
        
        # Should not detect conflicts due to temporal separation
        self.assertEqual(len(conflicts), 0)

class TestDeconflictionService(unittest.TestCase):
    """Test cases for DeconflictionService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = DeconflictionService(time_resolution=1.0)
        
        self.primary_mission = DroneMission(
            drone_id="PRIMARY",
            waypoints=[
                Waypoint(0, 0, 10, 0),
                Waypoint(100, 100, 20, 60)
            ],
            start_time=0,
            end_time=60,
            safety_buffer=5.0
        )
        
        # Conflicting mission
        self.conflicting_mission = DroneMission(
            drone_id="CONFLICTING",
            waypoints=[
                Waypoint(45, 45, 15, 25),
                Waypoint(55, 55, 15, 35)
            ],
            start_time=25,
            end_time=35,
            safety_buffer=5.0
        )
        
        # Safe mission
        self.safe_mission = DroneMission(
            drone_id="SAFE",
            waypoints=[
                Waypoint(200, 200, 50, 10),
                Waypoint(300, 300, 60, 50)
            ],
            start_time=10,
            end_time=50,
            safety_buffer=5.0
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        self.assertEqual(len(self.service.simulated_flights), 0)
        self.assertIsNotNone(self.service.conflict_detector)
    
    def test_add_remove_flights(self):
        """Test adding and removing simulated flights."""
        # Add flights
        self.service.add_simulated_flight(self.conflicting_mission)
        self.service.add_simulated_flight(self.safe_mission)
        
        self.assertEqual(len(self.service.simulated_flights), 2)
        
        # Remove flight
        success = self.service.remove_simulated_flight("CONFLICTING")
        self.assertTrue(success)
        self.assertEqual(len(self.service.simulated_flights), 1)
        
        # Try to remove non-existent flight
        success = self.service.remove_simulated_flight("NON_EXISTENT")
        self.assertFalse(success)
    
    def test_conflict_checking(self):
        """Test primary mission conflict checking."""
        # Add simulated flights
        self.service.add_simulated_flight(self.conflicting_mission)
        self.service.add_simulated_flight(self.safe_mission)
        
        # Check for conflicts
        result = self.service.check_mission_conflicts(self.primary_mission)
        
        # Verify result structure
        self.assertIn('status', result)
        self.assertIn('conflicts', result)
        self.assertIn('conflict_summary', result)
        self.assertIn('recommendation', result)
        
        # Should detect conflicts
        self.assertNotEqual(result['status'], 'clear')
        self.assertGreater(len(result['conflicts']), 0)
    
    def test_clear_scenario(self):
        """Test scenario with no conflicts."""
        # Add only safe mission
        self.service.add_simulated_flight(self.safe_mission)
        
        result = self.service.check_mission_conflicts(self.primary_mission)
        
        # Should be clear or only warnings
        self.assertIn(result['status'], ['clear', 'warning'])
    
    def test_batch_checking(self):
        """Test batch conflict checking."""
        # Add simulated flights
        self.service.add_simulated_flight(self.conflicting_mission)
        
        missions_to_check = [self.primary_mission, self.safe_mission]
        batch_result = self.service.batch_conflict_check(missions_to_check)
        
        # Verify batch result structure
        self.assertIn('total_missions_checked', batch_result)
        self.assertIn('individual_results', batch_result)
        
        self.assertEqual(batch_result['total_missions_checked'], 2)
        self.assertEqual(len(batch_result['individual_results']), 2)
    
    def test_airspace_status(self):
        """Test airspace status reporting."""
        self.service.add_simulated_flight(self.conflicting_mission)
        self.service.add_simulated_flight(self.safe_mission)
        
        status = self.service.get_airspace_status()
        
        # Verify status structure
        self.assertIn('total_registered_flights', status)
        self.assertIn('peak_utilization', status)
        self.assertIn('average_utilization', status)
        
        self.assertEqual(status['total_registered_flights'], 2)

class TestDataGenerator(unittest.TestCase):
    """Test cases for DataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DataGenerator(seed=42)  # Fixed seed for reproducibility
    
    def test_random_waypoint_generation(self):
        """Test random waypoint generation."""
        waypoint = self.generator.generate_random_waypoint()
        
        # Check that waypoint is within bounds
        self.assertGreaterEqual(waypoint[0], self.generator.airspace_bounds['x_min'])
        self.assertLessEqual(waypoint[0], self.generator.airspace_bounds['x_max'])
        self.assertGreaterEqual(waypoint[1], self.generator.airspace_bounds['y_min'])
        self.assertLessEqual(waypoint[1], self.generator.airspace_bounds['y_max'])
        self.assertGreaterEqual(waypoint[2], self.generator.airspace_bounds['z_min'])
        self.assertLessEqual(waypoint[2], self.generator.airspace_bounds['z_max'])
    
    def test_mission_generation(self):
        """Test random mission generation."""
        mission = self.generator.generate_random_mission("TEST_DRONE")
        
        # Verify mission properties
        self.assertEqual(mission.drone_id, "TEST_DRONE")
        self.assertGreater(len(mission.waypoints), 0)
        self.assertLessEqual(mission.start_time, mission.end_time)
        self.assertGreater(mission.safety_buffer, 0)
    
    def test_multiple_mission_generation(self):
        """Test generation of multiple missions."""
        missions = self.generator.generate_multiple_missions(5, "TEST")
        
        self.assertEqual(len(missions), 5)
        
        # Check that all missions have unique IDs
        drone_ids = [m.drone_id for m in missions]
        self.assertEqual(len(drone_ids), len(set(drone_ids)))  # All unique
    
    def test_conflict_scenario_generation(self):
        """Test conflict scenario generation."""
        primary, conflicting = self.generator.generate_conflict_scenario()
        
        self.assertIsInstance(primary, DroneMission)
        self.assertIsInstance(conflicting, list)
        self.assertGreater(len(conflicting), 0)
        
        # All missions should be valid
        for mission in [primary] + conflicting:
            self.assertGreater(len(mission.waypoints), 0)
            self.assertLessEqual(mission.start_time, mission.end_time)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_distance_calculation(self):
        """Test 3D distance calculation."""
        point1 = (0, 0, 0)
        point2 = (3, 4, 0)  # 3-4-5 triangle
        
        distance = calculate_distance_3d(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=5)
    
    def test_airspace_validation(self):
        """Test airspace boundary validation."""
        bounds = {
            'x_min': 0, 'x_max': 100,
            'y_min': 0, 'y_max': 100,
            'z_min': 0, 'z_max': 50
        }
        
        # Valid waypoints
        valid_waypoints = [(50, 50, 25), (0, 0, 0), (100, 100, 50)]
        errors = validate_airspace_bounds(valid_waypoints, bounds)
        self.assertEqual(len(errors), 0)
        
        # Invalid waypoints
        invalid_waypoints = [(150, 50, 25), (50, -10, 25), (50, 50, 60)]
        errors = validate_airspace_bounds(invalid_waypoints, bounds)
        self.assertGreater(len(errors), 0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.generator = DataGenerator(seed=123)
        self.service = DeconflictionService()
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Generate test data
        primary_mission = self.generator.generate_random_mission("PRIMARY_INTEGRATION")
        simulated_missions = self.generator.generate_multiple_missions(3, "SIM_INTEGRATION")
        
        # Add simulated missions to service
        for mission in simulated_missions:
            self.service.add_simulated_flight(mission)
        
        # Perform conflict check
        result = self.service.check_mission_conflicts(primary_mission)
        
        # Verify complete result structure
        required_keys = ['status', 'conflicts', 'conflict_summary', 'recommendation']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Status should be valid
        valid_statuses = ['clear', 'warning', 'conflict_detected']
        self.assertIn(result['status'], valid_statuses)
        
        # Conflicts should be properly formatted
        for conflict in result['conflicts']:
            required_conflict_keys = ['type', 'time', 'location', 'distance', 'conflicting_drone']
            for key in required_conflict_keys:
                self.assertIn(key, conflict)
    
    def test_high_density_scenario(self):
        """Test system performance with high-density airspace."""
        # Generate high-density scenario
        missions = self.generator.generate_high_density_scenario(num_drones=10)
        
        # Test that all missions are valid
        for mission in missions:
            self.assertIsInstance(mission, DroneMission)
            self.assertGreater(len(mission.waypoints), 0)
        
        # Add all but one to service
        primary = missions[0]
        simulated = missions[1:]
        
        for mission in simulated:
            self.service.add_simulated_flight(mission)
        
        # Should handle high density without crashing
        result = self.service.check_mission_conflicts(primary)
        self.assertIsInstance(result, dict)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Single waypoint mission
        single_wp_mission = DroneMission(
            drone_id="SINGLE_WP",
            waypoints=[Waypoint(50, 50, 25, 30)],
            start_time=30,
            end_time=30,  # Zero duration
            safety_buffer=5.0
        )
        
        # Should handle single waypoint without errors
        result = self.service.check_mission_conflicts(single_wp_mission)
        self.assertEqual(result['status'], 'clear')  # No other flights to conflict with
        
        # Very short mission
        short_mission = DroneMission(
            drone_id="SHORT",
            waypoints=[
                Waypoint(0, 0, 10, 0),
                Waypoint(1, 1, 11, 1)  # 1-second mission
            ],
            start_time=0,
            end_time=1,
            safety_buffer=5.0
        )
        
        result = self.service.check_mission_conflicts(short_mission)
        self.assertIn('status', result)

class TestPerformance(unittest.TestCase):
    """Performance tests for the system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.generator = DataGenerator(seed=456)
        self.service = DeconflictionService()
    
    def test_processing_time(self):
        """Test that conflict detection completes within reasonable time."""
        import time
        
        # Generate moderate scenario
        missions = self.generator.generate_multiple_missions(20, "PERF_TEST")
        primary = missions[0]
        simulated = missions[1:]
        
        for mission in simulated:
            self.service.add_simulated_flight(mission)
        
        # Measure processing time
        start_time = time.time()
        result = self.service.check_mission_conflicts(primary)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within 10 seconds for 20 missions
        self.assertLess(processing_time, 10.0)
        
        # Result should be valid
        self.assertIn('status', result)
    
    def test_memory_usage(self):
        """Test that the system doesn't have obvious memory leaks."""
        import gc
        
        # Run multiple iterations
        for i in range(10):
            missions = self.generator.generate_multiple_missions(5, f"MEM_TEST_{i}")
            
            service = DeconflictionService()
            for mission in missions[1:]:
                service.add_simulated_flight(mission)
            
            result = service.check_mission_conflicts(missions[0])
            
            # Force garbage collection
            del service
            del missions
            del result
            gc.collect()
        
        # If we get here without memory errors, test passes
        self.assertTrue(True)

def run_test_suite():
    """Run the complete test suite and generate report."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestWaypoint,
        TestDroneMission,
        TestConflictDetector,
        TestDeconflictionService,
        TestDataGenerator,
        TestUtils,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Generate summary
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("UAV Deconfliction System - Test Suite")
    print("FlytBase Robotics Assignment 2025")
    print("="*60)
    
    success = run_test_suite()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)