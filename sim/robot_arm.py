"""
6-DOF空间机械臂模型
基于DH参数的运动学，支持正/逆运动学和雅可比矩阵计算
"""
import numpy as np
from typing import Optional, Tuple, List


class RobotArm:
    """6自由度空间机械臂"""

    def __init__(self, link_lengths: Optional[List[float]] = None,
                 base_position: Optional[np.ndarray] = None,
                 joint_max_velocity: float = 3.14,
                 joint_max_torque: float = 100.0):

        self.link_lengths = link_lengths or [1.0, 1.5, 1.2, 0.8, 0.5, 0.3]
        self.n_joints = len(self.link_lengths)
        self.base_position = base_position if base_position is not None else np.zeros(3)
        self.joint_max_velocity = joint_max_velocity
        self.joint_max_torque = joint_max_torque

        # 关节角度和速度
        self.joint_angles = np.zeros(self.n_joints)
        self.joint_velocities = np.zeros(self.n_joints)

        # 关节限位
        self.joint_limits = np.array([[-np.pi, np.pi]] * self.n_joints)

        # 末端执行器状态
        self.gripper_open = True
        self.gripper_type = "three_finger"  # three_finger / flexible_envelope / electromagnetic

        # DH参数 [a, alpha, d, theta_offset]
        self._setup_dh_params()

    def _setup_dh_params(self):
        """设置DH参数 - 简化的6DOF构型"""
        L = self.link_lengths
        self.dh_params = np.array([
            [0,      np.pi/2, L[0], 0],  # 关节1: 基座旋转
            [L[1],   0,       0,    0],   # 关节2: 肩部
            [L[2],   0,       0,    0],   # 关节3: 肘部
            [0,      np.pi/2, 0,    0],   # 关节4: 腕部旋转
            [0,     -np.pi/2, L[3], 0],   # 关节5: 腕部俯仰
            [0,      0,       L[4]+L[5], 0],  # 关节6: 末端旋转
        ])

    def dh_transform(self, a, alpha, d, theta) -> np.ndarray:
        """DH变换矩阵"""
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])

    def forward_kinematics(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """正运动学 - 返回末端位姿 4x4矩阵"""
        if joint_angles is None:
            joint_angles = self.joint_angles

        T = np.eye(4)
        T[:3, 3] = self.base_position

        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(a, alpha, d, theta)

        return T

    def get_tip_position(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """获取末端执行器位置"""
        T = self.forward_kinematics(joint_angles)
        return T[:3, 3]

    def get_joint_positions(self, joint_angles: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """获取所有关节位置（用于可视化）"""
        if joint_angles is None:
            joint_angles = self.joint_angles

        positions = [self.base_position.copy()]
        T = np.eye(4)
        T[:3, 3] = self.base_position

        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(a, alpha, d, theta)
            positions.append(T[:3, 3].copy())

        return positions

    def jacobian(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """计算雅可比矩阵 (数值方法)"""
        if joint_angles is None:
            joint_angles = self.joint_angles

        J = np.zeros((3, self.n_joints))
        eps = 1e-6
        p0 = self.get_tip_position(joint_angles)

        for i in range(self.n_joints):
            q_plus = joint_angles.copy()
            q_plus[i] += eps
            p_plus = self.get_tip_position(q_plus)
            J[:, i] = (p_plus - p0) / eps

        return J

    def inverse_kinematics_step(self, target_pos: np.ndarray,
                                 step_size: float = 0.5) -> np.ndarray:
        """逆运动学 - 单步雅可比伪逆法"""
        current_pos = self.get_tip_position()
        error = target_pos - current_pos

        J = self.jacobian()

        # 阻尼最小二乘 (DLS) 避免奇异
        damping = 0.01
        JJT = J @ J.T + damping**2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error) * step_size

        # 限制关节速度
        max_dq = self.joint_max_velocity * 0.01  # 假设dt=0.01
        scale = np.max(np.abs(dq) / max_dq)
        if scale > 1.0:
            dq /= scale

        new_angles = self.joint_angles + dq

        # 关节限位
        for i in range(self.n_joints):
            new_angles[i] = np.clip(new_angles[i],
                                     self.joint_limits[i, 0],
                                     self.joint_limits[i, 1])

        return new_angles

    def move_to_target(self, target_pos: np.ndarray, dt: float = 0.01) -> bool:
        """向目标位置移动一步，返回是否到达"""
        new_angles = self.inverse_kinematics_step(target_pos)
        self.joint_velocities = (new_angles - self.joint_angles) / dt
        self.joint_angles = new_angles

        current_pos = self.get_tip_position()
        distance = np.linalg.norm(target_pos - current_pos)
        return distance < 0.05

    def get_tip_velocity(self) -> np.ndarray:
        """获取末端执行器速度"""
        J = self.jacobian()
        return J @ self.joint_velocities

    def close_gripper(self):
        self.gripper_open = False

    def open_gripper(self):
        self.gripper_open = True

    def check_singularity(self) -> float:
        """检查奇异度 - 返回雅可比矩阵的最小奇异值"""
        J = self.jacobian()
        singular_values = np.linalg.svd(J, compute_uv=False)
        return singular_values[-1]

    def get_manipulability(self) -> float:
        """可操作度指标"""
        J = self.jacobian()
        return np.sqrt(max(0, np.linalg.det(J @ J.T)))

    def reset(self, joint_angles: Optional[np.ndarray] = None):
        """重置机械臂"""
        if joint_angles is not None:
            self.joint_angles = joint_angles.copy()
        else:
            self.joint_angles = np.zeros(self.n_joints)
        self.joint_velocities = np.zeros(self.n_joints)
        self.gripper_open = True
