import numpy as np
from typing import List, Tuple, Dict
import math

class TrackingMetrics:
    def __init__(self):
        # Lists to store per-frame metrics
        self.translation_errors: List[float] = []
        self.rotation_errors: List[float] = []
        self.feature_counts: List[int] = []
        self.inlier_ratios: List[float] = []
        
        # Cumulative metrics
        self.total_translation_error = 0.0
        self.total_rotation_error = 0.0
        self.total_features = 0
        self.total_inliers = 0
        self.frame_count = 0
        
        # Ground truth and estimated trajectories
        self.gt_trajectory: List[np.ndarray] = []
        self.est_trajectory: List[np.ndarray] = []
        
    def update_metrics(self, 
                      gt_position: np.ndarray,
                      est_position: np.ndarray,
                      gt_yaw: float,
                      est_yaw: float,
                      num_features: int,
                      num_inliers: int):
        """
        Update metrics with new frame data
        
        Args:
            gt_position: Ground truth position (x, y)
            est_position: Estimated position (x, y)
            gt_yaw: Ground truth yaw angle in radians
            est_yaw: Estimated yaw angle in radians
            num_features: Number of features detected
            num_inliers: Number of inlier matches
        """
        # Store trajectories
        self.gt_trajectory.append(gt_position)
        self.est_trajectory.append(est_position)
        
        # Calculate translation error (in pixels)
        # Note: est_position is already in the same coordinate system as gt_position
        translation_error = np.linalg.norm(gt_position - est_position)
        self.translation_errors.append(translation_error)
        self.total_translation_error += translation_error
        
        # Calculate rotation error (in degrees)
        # Normalize angles to [-pi, pi] before comparison
        gt_yaw_norm = math.atan2(math.sin(gt_yaw), math.cos(gt_yaw))
        est_yaw_norm = math.atan2(math.sin(est_yaw), math.cos(est_yaw))
        rotation_error = math.degrees(abs(gt_yaw_norm - est_yaw_norm))
        # Take the smaller angle difference
        if rotation_error > 180:
            rotation_error = 360 - rotation_error
        self.rotation_errors.append(rotation_error)
        self.total_rotation_error += rotation_error
        
        # Update feature metrics
        self.feature_counts.append(num_features)
        self.total_features += num_features
        
        inlier_ratio = num_inliers / num_features if num_features > 0 else 0
        self.inlier_ratios.append(inlier_ratio)
        self.total_inliers += num_inliers
        
        self.frame_count += 1
    
    def get_absolute_trajectory_error(self) -> float:
        """Calculate Absolute Trajectory Error (ATE)"""
        if not self.gt_trajectory or not self.est_trajectory:
            return 0.0
        
        gt_array = np.array(self.gt_trajectory)
        est_array = np.array(self.est_trajectory)
        
        # Calculate RMSE directly since positions are already in the same coordinate system
        squared_errors = np.sum((gt_array - est_array) ** 2, axis=1)
        return np.sqrt(np.mean(squared_errors))
    
    def get_relative_pose_error(self) -> float:
        """Calculate Relative Pose Error (RPE)"""
        if len(self.gt_trajectory) < 2 or len(self.est_trajectory) < 2:
            return 0.0
        
        gt_relative_poses = []
        est_relative_poses = []
        
        for i in range(1, len(self.gt_trajectory)):
            # Calculate relative poses
            gt_relative = self.gt_trajectory[i] - self.gt_trajectory[i-1]
            est_relative = self.est_trajectory[i] - self.est_trajectory[i-1]
            
            gt_relative_poses.append(gt_relative)
            est_relative_poses.append(est_relative)
        
        # Calculate RMSE of relative poses
        squared_errors = np.sum((np.array(gt_relative_poses) - np.array(est_relative_poses)) ** 2, axis=1)
        return np.sqrt(np.mean(squared_errors))
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical metrics"""
        if self.frame_count == 0:
            return {
                "mean_translation_error": 0.0,
                "max_translation_error": 0.0,
                "mean_rotation_error": 0.0,
                "max_rotation_error": 0.0,
                "mean_feature_count": 0.0,
                "mean_inlier_ratio": 0.0,
                "ate": 0.0,
                "rpe": 0.0
            }
        
        return {
            "mean_translation_error": self.total_translation_error / self.frame_count,
            "max_translation_error": max(self.translation_errors),
            "mean_rotation_error": self.total_rotation_error / self.frame_count,
            "max_rotation_error": max(self.rotation_errors),
            "mean_feature_count": self.total_features / self.frame_count,
            "mean_inlier_ratio": self.total_inliers / self.total_features if self.total_features > 0 else 0,
            "ate": self.get_absolute_trajectory_error(),
            "rpe": self.get_relative_pose_error()
        }
    
    def print_metrics(self):
        """Print all metrics in a readable format"""
        stats = self.get_statistics()
        print("\n=== Tracking Metrics ===")
        print(f"Total Frames: {self.frame_count}")
        print("\nTranslation Errors:")
        print(f"  Mean: {stats['mean_translation_error']:.2f} pixels")
        print(f"  Max: {stats['max_translation_error']:.2f} pixels")
        print("\nRotation Errors:")
        print(f"  Mean: {stats['mean_rotation_error']:.2f} degrees")
        print(f"  Max: {stats['max_rotation_error']:.2f} degrees")
        print("\nFeature Tracking:")
        print(f"  Mean Features: {stats['mean_feature_count']:.1f}")
        print(f"  Mean Inlier Ratio: {stats['mean_inlier_ratio']:.2%}")
        print("\nTrajectory Errors:")
        print(f"  Absolute Trajectory Error (ATE): {stats['ate']:.2f} pixels")
        print(f"  Relative Pose Error (RPE): {stats['rpe']:.2f} pixels")

