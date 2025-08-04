"""
Visualization Module

This module provides comprehensive visualization capabilities for the UAV
deconfliction system, including 2D/3D trajectory plots and 4D animations.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Tuple
import colorsys

from .drone_mission import DroneMission

class Visualizer:
    """Main visualization class for drone trajectories and conflicts."""
    
    def __init__(self):
        self.color_palette = self._generate_color_palette(20)
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _generate_color_palette(self, n_colors: int) -> List[str]:
        """Generate a diverse color palette for multiple drones."""
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
        return colors
    
    def plot_2d_trajectories(self, primary_mission: DroneMission, 
                           simulated_missions: List[DroneMission],
                           conflicts: List[Dict[str, Any]]) -> plt.Figure:
        """
        Create a 2D top-down view of all trajectories with conflict markers.
        
        Args:
            primary_mission: Primary drone mission
            simulated_missions: List of simulated drone missions
            conflicts: List of detected conflicts
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot primary mission
        primary_trajectory = primary_mission.get_trajectory_points(0.5)
        if primary_trajectory:
            x_coords = [point[0] for point in primary_trajectory]
            y_coords = [point[1] for point in primary_trajectory]
            
            ax.plot(x_coords, y_coords, 'r-', linewidth=3, label=f'PRIMARY: {primary_mission.drone_id}', zorder=10)
            ax.plot(x_coords[0], y_coords[0], 'ro', markersize=10, label='Start', zorder=11)
            ax.plot(x_coords[-1], y_coords[-1], 'rs', markersize=10, label='End', zorder=11)
            
            # Plot waypoints
            for wp in primary_mission.waypoints:
                ax.plot(wp.x, wp.y, 'r^', markersize=8, zorder=12)
        
        # Plot simulated missions
        for i, mission in enumerate(simulated_missions):
            color = self.color_palette[i % len(self.color_palette)]
            trajectory = mission.get_trajectory_points(0.5)
            
            if trajectory:
                x_coords = [point[0] for point in trajectory]
                y_coords = [point[1] for point in trajectory]
                
                ax.plot(x_coords, y_coords, '--', color=color, linewidth=2, 
                       label=f'SIM: {mission.drone_id}', alpha=0.7)
                ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=6)
                ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=6)
        
        # Plot conflicts
        conflict_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'lightcoral', 'WARNING': 'blue'}
        
        for conflict in conflicts:
            location = conflict['location']
            severity = conflict.get('severity', 'LOW')
            color = conflict_colors.get(severity, 'gray')
            
            # Draw conflict zone
            circle = plt.Circle((location[0], location[1]), 
                              conflict.get('required_separation', 10)/2, 
                              color=color, alpha=0.3, zorder=5)
            ax.add_patch(circle)
            
            # Mark conflict center
            ax.plot(location[0], location[1], 'x', color=color, markersize=12, 
                   markeredgewidth=3, zorder=15)
        
        # Formatting
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('UAV Trajectory Analysis - Top View', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add conflict summary
        conflict_text = f"Conflicts Detected: {len(conflicts)}"
        if conflicts:
            severity_counts = {}
            for conflict in conflicts:
                severity = conflict.get('severity', 'LOW')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            conflict_details = []
            for severity, count in severity_counts.items():
                conflict_details.append(f"{severity}: {count}")
            
            conflict_text += f"\n{', '.join(conflict_details)}"
        
        ax.text(0.02, 0.98, conflict_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_3d_trajectories(self, primary_mission: DroneMission,
                           simulated_missions: List[DroneMission],
                           conflicts: List[Dict[str, Any]]) -> plt.Figure:
        """
        Create a 3D visualization of all trajectories with conflict markers.
        
        Args:
            primary_mission: Primary drone mission
            simulated_missions: List of simulated drone missions
            conflicts: List of detected conflicts
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot primary mission
        primary_trajectory = primary_mission.get_trajectory_points(0.5)
        if primary_trajectory:
            x_coords = [point[0] for point in primary_trajectory]
            y_coords = [point[1] for point in primary_trajectory]
            z_coords = [point[2] for point in primary_trajectory]
            
            ax.plot(x_coords, y_coords, z_coords, 'r-', linewidth=4, 
                   label=f'PRIMARY: {primary_mission.drone_id}', zorder=10)
            ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                      c='red', s=100, marker='o', label='Start', zorder=11)
            ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                      c='red', s=100, marker='s', label='End', zorder=11)
            
            # Plot waypoints
            for wp in primary_mission.waypoints:
                ax.scatter(wp.x, wp.y, wp.z, c='red', s=60, marker='^', zorder=12)
        
        # Plot simulated missions
        for i, mission in enumerate(simulated_missions):
            color = self.color_palette[i % len(self.color_palette)]
            trajectory = mission.get_trajectory_points(0.5)
            
            if trajectory:
                x_coords = [point[0] for point in trajectory]
                y_coords = [point[1] for point in trajectory]
                z_coords = [point[2] for point in trajectory]
                
                ax.plot(x_coords, y_coords, z_coords, '--', color=color, linewidth=2,
                       label=f'SIM: {mission.drone_id}', alpha=0.7)
                ax.scatter(x_coords[0], y_coords[0], z_coords[0], 
                          c=color, s=50, marker='o')
                ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                          c=color, s=50, marker='s')
        
        # Plot conflicts as 3D spheres
        conflict_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'lightcoral', 'WARNING': 'blue'}
        
        for conflict in conflicts:
            location = conflict['location']
            severity = conflict.get('severity', 'LOW')
            color = conflict_colors.get(severity, 'gray')
            radius = conflict.get('required_separation', 10) / 2
            
            # Create sphere for conflict zone
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = location[0] + radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = location[1] + radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = location[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color=color)
            
            # Mark conflict center
            ax.scatter(location[0], location[1], location[2], 
                      c=color, s=200, marker='x', linewidth=4, zorder=15)
        
        # Formatting
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        ax.set_title('UAV Trajectory Analysis - 3D View', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio
        all_trajectories = [primary_trajectory] + [m.get_trajectory_points(0.5) for m in simulated_missions]
        all_points = []
        for traj in all_trajectories:
            if traj:
                all_points.extend(traj)
        
        if all_points:
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            z_coords = [p[2] for p in all_points]
            
            max_range = max(max(x_coords) - min(x_coords),
                           max(y_coords) - min(y_coords),
                           max(z_coords) - min(z_coords)) / 2
            
            mid_x = (max(x_coords) + min(x_coords)) / 2
            mid_y = (max(y_coords) + min(y_coords)) / 2
            mid_z = (max(z_coords) + min(z_coords)) / 2
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig
    
    def create_4d_animation(self, primary_mission: DroneMission,
                          simulated_missions: List[DroneMission],
                          conflicts: List[Dict[str, Any]]) -> go.Figure:
        """
        Create an interactive 4D (3D + time) animation using Plotly.
        
        Args:
            primary_mission: Primary drone mission
            simulated_missions: List of simulated drone missions
            conflicts: List of detected conflicts
            
        Returns:
            Plotly Figure with animation
        """
        # Determine time range
        all_missions = [primary_mission] + simulated_missions
        start_time = min(m.start_time for m in all_missions)
        end_time = max(m.end_time for m in all_missions)
        
        time_steps = np.arange(start_time, end_time + 1, 1.0)
        
        # Create frames for animation
        frames = []
        
        for t in time_steps:
            frame_data = []
            
            # Primary mission
            primary_pos = primary_mission.get_position_at_time(t)
            if primary_pos is not None:
                # Current position
                frame_data.append(go.Scatter3d(
                    x=[primary_pos[0]], y=[primary_pos[1]], z=[primary_pos[2]],
                    mode='markers', marker=dict(size=12, color='red'),
                    name=f'PRIMARY: {primary_mission.drone_id}',
                    showlegend=True
                ))
                
                # Trail (past positions)
                trail_times = np.arange(max(start_time, t-10), t+0.1, 0.5)
                trail_positions = []
                for trail_t in trail_times:
                    trail_pos = primary_mission.get_position_at_time(trail_t)
                    if trail_pos is not None:
                        trail_positions.append(trail_pos)
                
                if len(trail_positions) > 1:
                    trail_x = [pos[0] for pos in trail_positions]
                    trail_y = [pos[1] for pos in trail_positions]
                    trail_z = [pos[2] for pos in trail_positions]
                    
                    frame_data.append(go.Scatter3d(
                        x=trail_x, y=trail_y, z=trail_z,
                        mode='lines', line=dict(color='red', width=6),
                        name='Primary Trail', showlegend=False
                    ))
            
            # Simulated missions
            for i, mission in enumerate(simulated_missions):
                color = self.color_palette[i % len(self.color_palette)]
                sim_pos = mission.get_position_at_time(t)
                
                if sim_pos is not None:
                    # Current position
                    frame_data.append(go.Scatter3d(
                        x=[sim_pos[0]], y=[sim_pos[1]], z=[sim_pos[2]],
                        mode='markers', marker=dict(size=8, color=color),
                        name=f'SIM: {mission.drone_id}',
                        showlegend=True
                    ))
                    
                    # Trail
                    trail_times = np.arange(max(start_time, t-10), t+0.1, 0.5)
                    trail_positions = []
                    for trail_t in trail_times:
                        trail_pos = mission.get_position_at_time(trail_t)
                        if trail_pos is not None:
                            trail_positions.append(trail_pos)
                    
                    if len(trail_positions) > 1:
                        trail_x = [pos[0] for pos in trail_positions]
                        trail_y = [pos[1] for pos in trail_positions]
                        trail_z = [pos[2] for pos in trail_positions]
                        
                        frame_data.append(go.Scatter3d(
                            x=trail_x, y=trail_y, z=trail_z,
                            mode='lines', line=dict(color=color, width=3),
                            name=f'{mission.drone_id} Trail', showlegend=False
                        ))
            
            # Add conflicts active at this time
            active_conflicts = [c for c in conflicts if abs(c['time'] - t) < 2.0]
            for conflict in active_conflicts:
                location = conflict['location']
                severity = conflict.get('severity', 'LOW')
                conflict_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'pink', 'WARNING': 'blue'}
                color = conflict_colors.get(severity, 'gray')
                
                frame_data.append(go.Scatter3d(
                    x=[location[0]], y=[location[1]], z=[location[2]],
                    mode='markers', marker=dict(size=20, color=color, symbol='x'),
                    name=f'Conflict ({severity})', showlegend=False
                ))
            
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        # Create initial figure
        fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
        
        # Add animation controls
        fig.update_layout(
            title="4D UAV Trajectory Animation (3D + Time)",
            scene=dict(
                xaxis_title="X Position (m)",
                yaxis_title="Y Position (m)",
                zaxis_title="Z Position (m)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True, "transition": {"duration": 300}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], {"frame": {"duration": 300, "redraw": True},
                                           "mode": "immediate", "transition": {"duration": 300}}],
                        "label": f"{f.name}s",
                        "method": "animate"
                    } for f in frames
                ]
            }]
        )
        
        return fig
    
    def plot_conflict_timeline(self, primary_mission: DroneMission,
                             simulated_missions: List[DroneMission],
                             conflicts: List[Dict[str, Any]]) -> plt.Figure:
        """
        Create a timeline visualization showing when conflicts occur.
        
        Args:
            primary_mission: Primary drone mission
            simulated_missions: List of simulated drone missions
            conflicts: List of detected conflicts
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 2])
        
        # Timeline of active missions
        all_missions = [primary_mission] + simulated_missions
        
        for i, mission in enumerate(all_missions):
            color = 'red' if mission == primary_mission else self.color_palette[i % len(self.color_palette)]
            linewidth = 4 if mission == primary_mission else 2
            
            ax1.barh(i, mission.end_time - mission.start_time, 
                    left=mission.start_time, height=0.6,
                    color=color, alpha=0.7, 
                    label=mission.drone_id)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Missions')
        ax1.set_title('Mission Timeline')
        ax1.set_yticks(range(len(all_missions)))
        ax1.set_yticklabels([m.drone_id for m in all_missions])
        ax1.grid(True, alpha=0.3)
        
        # Conflict timeline
        if conflicts:
            conflict_times = [c['time'] for c in conflicts]
            conflict_severities = [c.get('severity', 'LOW') for c in conflicts]
            
            severity_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'lightcoral', 'WARNING': 'blue'}
            colors = [severity_colors.get(sev, 'gray') for sev in conflict_severities]
            
            # Plot conflicts as scatter points
            y_positions = range(len(conflicts))
            ax2.scatter(conflict_times, y_positions, c=colors, s=100, alpha=0.8, zorder=5)
            
            # Add conflict details
            for i, conflict in enumerate(conflicts):
                ax2.annotate(f"{conflict['conflicting_drone']}\n{conflict['distance']:.1f}m", 
                           (conflict['time'], i),
                           xytext=(10, 0), textcoords='offset points',
                           va='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Conflicts')
        ax2.set_title('Conflict Timeline')
        ax2.grid(True, alpha=0.3)
        
        if conflicts:
            ax2.set_yticks(range(len(conflicts)))
            ax2.set_yticklabels([f"Conflict {i+1}" for i in range(len(conflicts))])
        
        plt.tight_layout()
        return fig