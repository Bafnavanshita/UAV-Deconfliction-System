#!/usr/bin/env python3
"""
UAV Strategic Deconfliction System - Main Entry Point
FlytBase Robotics Assignment 2025

This module serves as the main entry point for the UAV deconfliction system,
demonstrating various scenarios and generating visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
import os

from src.drone_mission import DroneMission, Waypoint
from src.deconfliction_service import DeconflictionService
from src.visualization import Visualizer
from src.data_generator import DataGenerator
from src.utils import setup_logging, save_results

def create_sample_missions() -> tuple[DroneMission, List[DroneMission]]:
    """Create sample drone missions for demonstration."""
    
    # Primary mission waypoints
    primary_waypoints = [
        Waypoint(0, 0, 10, 0),      # Start at origin
        Waypoint(50, 50, 15, 30),   # Move to (50,50) at altitude 15
        Waypoint(100, 25, 20, 60),  # Move to (100,25) at altitude 20
        Waypoint(150, 75, 12, 90),  # Move to (150,75) at altitude 12
        Waypoint(200, 100, 18, 120) # End at (200,100) at altitude 18
    ]
    
    primary_mission = DroneMission(
        drone_id="PRIMARY_001",
        waypoints=primary_waypoints,
        start_time=0,
        end_time=150,
        safety_buffer=5.0
    )
    
    # Create conflicting simulated missions
    simulated_missions = []
    
    # Mission 1: Spatial conflict
    conflict_waypoints_1 = [
        Waypoint(30, 30, 14, 20),
        Waypoint(70, 70, 16, 50),
        Waypoint(120, 50, 19, 80)
    ]
    simulated_missions.append(DroneMission(
        drone_id="SIM_001", 
        waypoints=conflict_waypoints_1,
        start_time=15,
        end_time=100,
        safety_buffer=5.0
    ))
    
    # Mission 2: No conflict
    safe_waypoints = [
        Waypoint(0, 200, 25, 10),
        Waypoint(50, 250, 30, 40),
        Waypoint(100, 300, 35, 70)
    ]
    simulated_missions.append(DroneMission(
        drone_id="SIM_002",
        waypoints=safe_waypoints,
        start_time=0,
        end_time=80,
        safety_buffer=5.0
    ))
    
    # Mission 3: Temporal conflict
    temporal_conflict_waypoints = [
        Waypoint(180, 90, 17, 100),
        Waypoint(220, 120, 20, 130),
        Waypoint(250, 150, 15, 160)
    ]
    simulated_missions.append(DroneMission(
        drone_id="SIM_003",
        waypoints=temporal_conflict_waypoints,
        start_time=80,
        end_time=180,
        safety_buffer=5.0
    ))
    
    return primary_mission, simulated_missions

def run_conflict_detection_demo():
    """Run the main conflict detection demonstration."""
    print("=" * 60)
    print("UAV STRATEGIC DECONFLICTION SYSTEM")
    print("FlytBase Robotics Assignment 2025")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting UAV Deconfliction System Demo")
    
    # Create missions
    primary_mission, simulated_missions = create_sample_missions()
    
    # Initialize deconfliction service
    deconfliction_service = DeconflictionService()
    
    # Add simulated missions to the service
    for mission in simulated_missions:
        deconfliction_service.add_simulated_flight(mission)
    
    print(f"\nPrimary Mission: {primary_mission.drone_id}")
    print(f"Time Window: {primary_mission.start_time}s - {primary_mission.end_time}s")
    print(f"Waypoints: {len(primary_mission.waypoints)}")
    print(f"Safety Buffer: {primary_mission.safety_buffer}m")
    
    print(f"\nSimulated Flights: {len(simulated_missions)}")
    for i, mission in enumerate(simulated_missions):
        print(f"  {i+1}. {mission.drone_id} ({len(mission.waypoints)} waypoints)")
    
    # Perform conflict check
    print("\n" + "="*40)
    print("PERFORMING CONFLICT DETECTION...")
    print("="*40)
    
    result = deconfliction_service.check_mission_conflicts(primary_mission)
    
    # Display results
    print(f"\nStatus: {result['status'].upper()}")
    
    if result['status'] == 'conflict_detected':
        print(f"Conflicts Found: {len(result['conflicts'])}")
        for i, conflict in enumerate(result['conflicts']):
            print(f"\nConflict {i+1}:")
            print(f"  Type: {conflict['type']}")
            print(f"  Location: ({conflict['location'][0]:.1f}, {conflict['location'][1]:.1f}, {conflict['location'][2]:.1f})")
            print(f"  Time: {conflict['time']:.1f}s")
            print(f"  Distance: {conflict['distance']:.1f}m")
            print(f"  Conflicting Drone: {conflict['conflicting_drone']}")
            print(f"  Description: {conflict['description']}")
    else:
        print("No conflicts detected - Mission is CLEAR for execution!")
    
    # Generate visualizations
    print("\n" + "="*40)
    print("GENERATING VISUALIZATIONS...")
    print("="*40)
    
    visualizer = Visualizer()
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # 2D Visualization
    print("Creating 2D trajectory plot...")
    fig_2d = visualizer.plot_2d_trajectories(primary_mission, simulated_missions, result['conflicts'])
    plt.savefig("output/trajectory_2d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3D Visualization
    print("Creating 3D trajectory plot...")
    fig_3d = visualizer.plot_3d_trajectories(primary_mission, simulated_missions, result['conflicts'])
    plt.savefig("output/trajectory_3d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4D Visualization (3D + Time animation)
    print("Creating 4D animation...")
    animation = visualizer.create_4d_animation(primary_mission, simulated_missions, result['conflicts'])
    animation.write_html("output/4d_animation.html")
    print("4D animation saved as HTML file")
    
    # Time-series conflict analysis
    print("Creating conflict timeline...")
    fig_timeline = visualizer.plot_conflict_timeline(primary_mission, simulated_missions, result['conflicts'])
    plt.savefig("output/conflict_timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    save_results(result, "output/conflict_analysis.json")
    
    print("\n" + "="*40)
    print("DEMONSTRATION COMPLETE!")
    print("="*40)
    print("Generated files:")
    print("  - output/trajectory_2d.png")
    print("  - output/trajectory_3d.png") 
    print("  - output/4d_animation.html")
    print("  - output/conflict_timeline.png")
    print("  - output/conflict_analysis.json")
    
    return result

if __name__ == "__main__":
    try:
        result = run_conflict_detection_demo()
        
        # Additional demo scenarios
        print("\n" + "="*60)
        print("RUNNING ADDITIONAL SCENARIOS...")
        print("="*60)
        
        # Generate multiple random scenarios
        data_generator = DataGenerator()
        
        for scenario_num in range(1, 4):
            print(f"\nScenario {scenario_num}:")
            
            # Generate random missions
            primary = data_generator.generate_random_mission(f"PRIMARY_{scenario_num:03d}")
            simulated = data_generator.generate_multiple_missions(5, f"SIM_{scenario_num}")
            
            # Run conflict detection
            service = DeconflictionService()
            for mission in simulated:
                service.add_simulated_flight(mission)
            
            result = service.check_mission_conflicts(primary)
            print(f"  Status: {result['status']}")
            print(f"  Conflicts: {len(result['conflicts'])}")
            
            # Save scenario results
            scenario_dir = f"output/scenario_{scenario_num}"
            os.makedirs(scenario_dir, exist_ok=True)
            
            visualizer = Visualizer()
            fig = visualizer.plot_3d_trajectories(primary, simulated, result['conflicts'])
            plt.savefig(f"{scenario_dir}/trajectory_3d.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            save_results(result, f"{scenario_dir}/analysis.json")
        
        print("\nAll scenarios completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()