"""
训练数据生成模块
通过仿真生成大量碎片运动轨迹用于训练LSTM预测模型
"""
import numpy as np
from typing import Tuple, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.debris import SpaceDebris


def generate_trajectory(linear_speed: float, angular_speed: float,
                        shape: str = "satellite", duration: float = 30.0,
                        dt: float = 0.01) -> np.ndarray:
    """
    生成单条碎片运动轨迹
    返回: (num_steps, 10) - [pos(3), vel(3), orient(4)]
    """
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)
    init_pos = direction * 10.0

    vel_dir = -direction + np.random.randn(3) * 0.2
    vel_dir /= np.linalg.norm(vel_dir)
    velocity = vel_dir * linear_speed

    ang_dir = np.random.randn(3)
    ang_dir /= np.linalg.norm(ang_dir)
    ang_vel = ang_dir * angular_speed

    debris = SpaceDebris(shape=shape, position=init_pos,
                         velocity=velocity, angular_velocity=ang_vel)

    num_steps = int(duration / dt)
    trajectory = np.zeros((num_steps, 10))

    for i in range(num_steps):
        trajectory[i, :3] = debris.position
        trajectory[i, 3:6] = debris.velocity
        trajectory[i, 6:10] = debris.orientation
        debris.step(dt)

    return trajectory


def generate_dataset(num_trajectories: int = 500,
                     seq_length: int = 30,
                     pred_horizon: int = 50,
                     speed_range: Tuple[float, float] = (0.1, 2.0),
                     ang_speed_range: Tuple[float, float] = (0.0, 0.524),
                     noise_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成训练数据集
    返回: X (N, seq_length, 10), Y (N, pred_horizon, 10)
    """
    X_list = []
    Y_list = []
    shapes = ["satellite", "rocket_stage", "fragment"]

    for t_idx in range(num_trajectories):
        speed = np.random.uniform(*speed_range)
        ang_speed = np.random.uniform(*ang_speed_range)
        shape = np.random.choice(shapes)

        traj = generate_trajectory(speed, ang_speed, shape)

        # 从轨迹中采样多个训练样本
        max_start = len(traj) - seq_length - pred_horizon
        if max_start <= 0:
            continue

        num_samples = min(20, max_start)
        starts = np.random.choice(max_start, num_samples, replace=False)

        for start in starts:
            x = traj[start:start + seq_length].copy()
            y = traj[start + seq_length:start + seq_length + pred_horizon].copy()

            # 添加观测噪声到输入
            x[:, :3] += np.random.normal(0, noise_std, (seq_length, 3))
            x[:, 3:6] += np.random.normal(0, noise_std * 0.5, (seq_length, 3))

            X_list.append(x)
            Y_list.append(y)

        if (t_idx + 1) % 100 == 0:
            print(f"  Generated {t_idx + 1}/{num_trajectories} trajectories, "
                  f"{len(X_list)} samples")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)

    return X, Y


def compute_normalization(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算归一化参数"""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.clip(std, 1e-6, None)
    return mean, std
