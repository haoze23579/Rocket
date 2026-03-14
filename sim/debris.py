"""
太空垃圾模型 - 模拟不同类型空间碎片的运动
支持三轴旋转叠加的翻滚运动，不同形状和尺寸
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DebrisState:
    """碎片状态"""
    position: np.ndarray        # [x, y, z] 位置
    velocity: np.ndarray        # [vx, vy, vz] 线速度
    orientation: np.ndarray     # [qw, qx, qy, qz] 四元数姿态
    angular_velocity: np.ndarray  # [wx, wy, wz] 角速度
    timestamp: float = 0.0


class SpaceDebris:
    """太空垃圾模型"""

    SHAPE_PARAMS = {
        "satellite": {"size": (2.0, 1.5, 1.0), "mass": 500.0, "grasp_radius": 0.8},
        "rocket_stage": {"size": (4.0, 1.2, 1.2), "mass": 1000.0, "grasp_radius": 0.6},
        "fragment": {"size": (0.5, 0.4, 0.3), "mass": 10.0, "grasp_radius": 0.25},
    }

    def __init__(self, shape: str = "satellite",
                 position: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None,
                 angular_velocity: Optional[np.ndarray] = None):
        if shape not in self.SHAPE_PARAMS:
            raise ValueError(f"Unknown shape: {shape}. Choose from {list(self.SHAPE_PARAMS.keys())}")

        self.shape = shape
        self.params = self.SHAPE_PARAMS[shape]
        self.size = np.array(self.params["size"])
        self.mass = self.params["mass"]
        self.grasp_radius = self.params["grasp_radius"]

        # 惯性张量 (简化为长方体)
        m = self.mass
        sx, sy, sz = self.size
        self.inertia = np.diag([
            m * (sy**2 + sz**2) / 12,
            m * (sx**2 + sz**2) / 12,
            m * (sx**2 + sy**2) / 12
        ])
        self.inertia_inv = np.linalg.inv(self.inertia)

        # 初始状态
        self.position = position if position is not None else np.array([10.0, 0.0, 0.0])
        self.velocity = velocity if velocity is not None else np.array([-0.5, 0.1, 0.05])
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数
        self.angular_velocity = angular_velocity if angular_velocity is not None else np.array([0.1, 0.05, 0.02])

        self.time = 0.0
        self.trajectory_history: List[DebrisState] = []

        # 抓取点 (物体坐标系下)
        self._compute_grasp_points()

    def _compute_grasp_points(self):
        """计算可抓取点位置"""
        sx, sy, sz = self.size / 2
        self.grasp_points = np.array([
            [sx, 0, 0], [-sx, 0, 0],
            [0, sy, 0], [0, -sy, 0],
            [0, 0, sz], [0, 0, -sz],
        ])

    def get_state(self) -> DebrisState:
        return DebrisState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            timestamp=self.time
        )

    def step(self, dt: float):
        """前进一个时间步 - 微重力下的自由运动"""
        # 记录历史
        self.trajectory_history.append(self.get_state())

        # 线性运动 (微重力，无外力)
        self.position += self.velocity * dt

        # 旋转运动 - 欧拉方程 (无外力矩时角动量守恒，但角速度可能变化)
        # τ = I·α + ω × (I·ω), τ=0
        L = self.inertia @ self.angular_velocity
        torque_gyro = -np.cross(self.angular_velocity, L)
        alpha = self.inertia_inv @ torque_gyro
        self.angular_velocity += alpha * dt

        # 四元数更新
        self.orientation = self._quaternion_integrate(self.orientation, self.angular_velocity, dt)

        self.time += dt

    def _quaternion_integrate(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """四元数积分更新姿态"""
        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-10:
            return q

        half_angle = omega_norm * dt / 2
        axis = omega / omega_norm
        dq = np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle)
        ])
        q_new = self._quaternion_multiply(q, dq)
        return q_new / np.linalg.norm(q_new)

    @staticmethod
    def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_rotation_matrix(self) -> np.ndarray:
        """四元数转旋转矩阵"""
        w, x, y, z = self.orientation
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    def get_world_grasp_points(self) -> np.ndarray:
        """获取世界坐标系下的抓取点"""
        R = self.get_rotation_matrix()
        return (R @ self.grasp_points.T).T + self.position

    def get_best_grasp_point(self, arm_tip_pos: np.ndarray) -> Tuple[np.ndarray, int]:
        """获取距离机械臂末端最近的抓取点"""
        world_points = self.get_world_grasp_points()
        distances = np.linalg.norm(world_points - arm_tip_pos, axis=1)
        best_idx = np.argmin(distances)
        return world_points[best_idx], best_idx

    def get_velocity_at_grasp_point(self, grasp_idx: int) -> np.ndarray:
        """获取抓取点的世界坐标系速度 (线速度 + 旋转引起的速度)"""
        R = self.get_rotation_matrix()
        r_world = R @ self.grasp_points[grasp_idx]
        v_rot = np.cross(self.angular_velocity, r_world)
        return self.velocity + v_rot
