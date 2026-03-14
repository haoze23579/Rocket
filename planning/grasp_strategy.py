"""
抓取策略模块
根据目标特性选择最优抓取策略和时机
"""
import numpy as np
from typing import Dict, Tuple, Optional


class GraspStrategy:
    """抓取策略决策"""

    # 不同末端执行器的特性
    GRIPPER_SPECS = {
        "three_finger": {
            "max_size": 1.5,       # 最大抓取尺寸
            "grasp_time": 0.3,     # 闭合时间(秒)
            "hold_force": 50.0,    # 保持力(N)
            "velocity_tolerance": 0.3,  # 速度容差(m/s)
        },
        "flexible_envelope": {
            "max_size": 3.0,
            "grasp_time": 0.5,
            "hold_force": 30.0,
            "velocity_tolerance": 0.5,
        },
        "electromagnetic": {
            "max_size": 5.0,
            "grasp_time": 0.1,
            "hold_force": 100.0,
            "velocity_tolerance": 0.8,
            "requires_metal": True,
        },
    }

    # 碎片形状与推荐夹爪的映射
    SHAPE_GRIPPER_MAP = {
        "satellite": ["flexible_envelope", "electromagnetic", "three_finger"],
        "rocket_stage": ["electromagnetic", "flexible_envelope"],
        "fragment": ["three_finger", "flexible_envelope"],
    }

    def __init__(self, config: dict):
        self.config = config
        self.success_threshold = config["grasp"]["success_threshold"]
        self.velocity_threshold = config["grasp"]["velocity_threshold"]

    def select_gripper(self, debris_shape: str, debris_size: float) -> str:
        """根据碎片特性选择最优夹爪"""
        candidates = self.SHAPE_GRIPPER_MAP.get(debris_shape, ["three_finger"])
        available = self.config["grasp"]["types"]

        for gripper in candidates:
            if gripper in available:
                spec = self.GRIPPER_SPECS[gripper]
                if debris_size <= spec["max_size"]:
                    return gripper

        return available[0]

    def should_grasp(self, distance: float, relative_velocity: float,
                     gripper_type: str, confidence: float = 1.0) -> Tuple[bool, str]:
        """
        判断是否应该执行抓取
        返回: (是否抓取, 原因)
        """
        spec = self.GRIPPER_SPECS.get(gripper_type, self.GRIPPER_SPECS["three_finger"])

        # 距离检查
        if distance > self.success_threshold * 2:
            return False, f"too_far (d={distance:.3f}m)"

        # 速度检查
        if relative_velocity > spec["velocity_tolerance"]:
            return False, f"too_fast (v={relative_velocity:.3f}m/s)"

        # 置信度检查
        if confidence < 0.3:
            return False, f"low_confidence ({confidence:.2f})"

        # 距离足够近且速度匹配
        if distance < self.success_threshold and relative_velocity < spec["velocity_tolerance"]:
            return True, "conditions_met"

        return False, "approaching"

    def compute_approach_vector(self, debris_position: np.ndarray,
                                 debris_velocity: np.ndarray,
                                 arm_tip_position: np.ndarray) -> np.ndarray:
        """
        计算最优接近方向
        考虑碎片运动方向，从侧面或追赶方向接近
        """
        # 碎片到机械臂的方向
        to_arm = arm_tip_position - debris_position
        to_arm_norm = to_arm / (np.linalg.norm(to_arm) + 1e-6)

        # 碎片运动方向
        vel_norm = debris_velocity / (np.linalg.norm(debris_velocity) + 1e-6)

        # 最优接近方向: 从碎片运动方向的侧面接近
        # 避免正面碰撞
        side = np.cross(vel_norm, to_arm_norm)
        side_norm = np.linalg.norm(side)
        if side_norm > 0.1:
            side /= side_norm
            approach = vel_norm * 0.7 + side * 0.3  # 追赶+侧面
        else:
            approach = vel_norm  # 从后方追赶

        return approach / np.linalg.norm(approach)
