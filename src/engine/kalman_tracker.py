"""
src/engine/kalman_tracker.py
============================
Industrial Kalman Filter for Multi-Object Worker Tracking (ByteTrack Core).
State space representation:
    x = [xc, yc, a, h, v_xc, v_yc, v_a, v_h]^T
where:
    (xc, yc) = center coordinates of bounding box
    a = aspect ratio (w / h)
    h = height of bounding box
    v_* = velocities of respective components
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space using a constant-velocity motion model.
    """

    def __init__(self):
        ndim = 4
        dt = 1.0

        # Motion model state transition matrix F
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Observation matrix H
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        measurement: bounding box coordinates (x_center, y_center, a, h) with a = w / h.
        Returns:
            mean: 8-dimensional state vector
            covariance: 8x8 covariance matrix
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        # Kalman Height Velocity Damping: clamp vertical height velocity |v_h| <= 0.15 * h
        # to prevent Kalman prediction explosion when box toggles between upper body and full body
        h = max(1.0, float(mean[3]))
        max_vh = 0.15 * h
        mean[7] = np.clip(mean[7], -max_vh, max_vh)

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T)) + innovation_cov
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter correction step.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Cholesky factorization for stable inversion
        chol_factor, lower = np.linalg.cholesky(projected_cov), True
        kalman_gain = np.linalg.solve(
            projected_cov,
            np.dot(covariance, self._update_mat.T).T
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

        # Kalman Height Velocity Damping: clamp vertical height velocity |v_h| <= 0.15 * h
        h = max(1.0, float(new_mean[3]))
        max_vh = 0.15 * h
        new_mean[7] = np.clip(new_mean[7], -max_vh, max_vh)

        return new_mean, new_covariance
