"""
视觉传感器模拟模块
模拟双目相机/RGB-D传感器，提供带噪声的目标观测
"""
import numpy as np
from typing import Optional, Dict
from .debris import SpaceDebris, DebrisState


class VisionSensor:
    """模拟视觉传感器"""

    def __init__(self, frame_rate: float = 30.0,
                 noise_std: float = 0.01,
                 max_range: float = 15.0,
                 fov: float = 90.0):
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.noise_std = noise_std
        self.max_range = max_range
        self.fov = np.radians(fov)

        self.last_capture_time = -1.0
        self.observation_buffer = []
        self.max_buffer_size = 300  # 保存最近10秒的观测

    def can_capture(self, current_time: float) -> bool:
        """检查是否到了采集时间"""
        return (current_time - self.last_capture_time) >= self.dt

    def observe(self, debris: SpaceDebris, current_time: float) -> Optional[Dict]:
        """
        观测目标碎片，返回带噪声的观测数据
        模拟真实传感器的测量误差
        """
        if not self.can_capture(current_time):
            return None

        self.last_capture_time = current_time

        # 检查是否在探测范围内
        distance = np.linalg.norm(debris.position)
        if distance > self.max_range:
            return None

        # 添加测量噪声 (距离越远噪声越大)
        noise_scale = self.noise_std * (1.0 + distance / self.max_range)

        position_noise = np.random.normal(0, noise_scale, 3)
        orientation_noise = np.random.normal(0, noise_scale * 0.1, 4)
        velocity_noise = np.random.normal(0, noise_scale * 0.5, 3)

        observation = {
            "timestamp": current_time,
            "position": debris.position + position_noise,
            "velocity": debris.velocity + velocity_noise,
            "orientation": debris.orientation + orientation_noise,
            "angular_velocity": debris.angular_velocity + np.random.normal(0, noise_scale * 0.3, 3),
            "distance": distance + np.random.normal(0, noise_scale),
            "shape_type": debris.shape,
            "confidence": max(0.0, min(1.0, 1.0 - distance / self.max_range + np.random.normal(0, 0.05))),
        }

        # 归一化四元数
        q = observation["orientation"]
        observation["orientation"] = q / np.linalg.norm(q)

        self.observation_buffer.append(observation)
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer.pop(0)

        return observation

    def get_observation_sequence(self, length: int = 30) -> Optional[np.ndarray]:
        """
        获取最近的观测序列，用于运动预测
        返回 shape: (length, 10) - [x,y,z, vx,vy,vz, qw,qx,qy,qz]
        """
        if len(self.observation_buffer) < length:
            return None

        recent = self.observation_buffer[-length:]
        sequence = np.zeros((length, 10))
        for i, obs in enumerate(recent):
            sequence[i, :3] = obs["position"]
            sequence[i, 3:6] = obs["velocity"]
            sequence[i, 6:10] = obs["orientation"]

        return sequence

    def reset(self):
        self.last_capture_time = -1.0
        self.observation_buffer.clear()
