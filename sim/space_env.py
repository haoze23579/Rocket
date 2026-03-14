"""
太空环境仿真主模块
整合碎片运动、机械臂控制、传感器观测
"""
import numpy as np
from typing import Optional, Dict, Tuple
from .debris import SpaceDebris
from .robot_arm import RobotArm
from .sensors import VisionSensor


class SpaceEnvironment:
    """太空环境仿真"""

    def __init__(self, config: dict):
        self.config = config
        self.dt = config["simulation"]["dt"]
        self.time = 0.0
        self.done = False
        self.success = False

        # 创建组件
        self.debris = None
        self.arm = None
        self.sensor = None

        self._setup_from_config(config)

        # 记录
        self.capture_distance_history = []
        self.arm_tip_history = []
        self.debris_pos_history = []

    def _setup_from_config(self, config: dict):
        arm_cfg = config["robot_arm"]
        self.arm = RobotArm(
            link_lengths=arm_cfg["link_lengths"],
            base_position=np.array(arm_cfg["base_position"]),
            joint_max_velocity=arm_cfg["joint_max_velocity"],
            joint_max_torque=arm_cfg["joint_max_torque"]
        )

        vis_cfg = config["vision"]
        self.sensor = VisionSensor(
            frame_rate=vis_cfg["frame_rate"],
            noise_std=vis_cfg["noise_std"],
            max_range=vis_cfg["max_range"],
            fov=vis_cfg["fov"]
        )

    def reset(self, linear_speed: Optional[float] = None,
              angular_speed: Optional[float] = None,
              shape: str = "satellite") -> Dict:
        """重置环境"""
        self.time = 0.0
        self.done = False
        self.success = False
        self.capture_distance_history.clear()
        self.arm_tip_history.clear()
        self.debris_pos_history.clear()

        dcfg = self.config["debris"]

        # 随机或指定速度
        if linear_speed is None:
            linear_speed = np.random.uniform(*dcfg["linear_velocity_range"])
        if angular_speed is None:
            angular_speed = np.random.uniform(*dcfg["angular_velocity_range"])

        # 碎片从随机方向接近
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        init_pos = direction * dcfg["initial_distance"]

        # 速度大致朝向原点（机械臂基座）
        vel_dir = -direction + np.random.randn(3) * 0.2
        vel_dir = vel_dir / np.linalg.norm(vel_dir)
        velocity = vel_dir * linear_speed

        # 随机角速度
        ang_dir = np.random.randn(3)
        ang_dir = ang_dir / np.linalg.norm(ang_dir)
        ang_vel = ang_dir * angular_speed

        self.debris = SpaceDebris(
            shape=shape,
            position=init_pos,
            velocity=velocity,
            angular_velocity=ang_vel
        )

        # 重置机械臂到初始位姿
        init_angles = np.array([0.0, -0.3, 0.6, 0.0, -0.3, 0.0])
        self.arm.reset(init_angles)
        self.sensor.reset()

        return self._get_info()

    def step(self, target_pos: Optional[np.ndarray] = None,
             close_gripper: bool = False) -> Tuple[Dict, bool, bool]:
        """
        仿真前进一步
        target_pos: 机械臂末端目标位置 (None则不移动)
        close_gripper: 是否闭合夹爪
        返回: (info, done, success)
        """
        # 碎片运动
        self.debris.step(self.dt)

        # 机械臂运动
        if target_pos is not None:
            self.arm.move_to_target(target_pos, self.dt)

        if close_gripper:
            self.arm.close_gripper()

        # 传感器观测
        observation = self.sensor.observe(self.debris, self.time)

        # 计算距离
        tip_pos = self.arm.get_tip_position()
        debris_pos = self.debris.position
        distance = np.linalg.norm(tip_pos - debris_pos)

        self.capture_distance_history.append(distance)
        self.arm_tip_history.append(tip_pos.copy())
        self.debris_pos_history.append(debris_pos.copy())

        # 检查抓取成功
        grasp_cfg = self.config["grasp"]
        if not self.arm.gripper_open and distance < grasp_cfg["success_threshold"]:
            tip_vel = self.arm.get_tip_velocity()
            debris_vel = self.debris.velocity
            rel_vel = np.linalg.norm(tip_vel - debris_vel)
            if rel_vel < grasp_cfg["velocity_threshold"]:
                self.success = True
                self.done = True

        # 检查碎片是否飞出范围
        if np.linalg.norm(debris_pos) > self.config["simulation"]["workspace_size"]:
            self.done = True

        # 超时检查 (30秒)
        if self.time > 30.0:
            self.done = True

        self.time += self.dt

        return self._get_info(), self.done, self.success

    def _get_info(self) -> Dict:
        tip_pos = self.arm.get_tip_position()
        debris_pos = self.debris.position if self.debris else np.zeros(3)
        distance = np.linalg.norm(tip_pos - debris_pos) if self.debris else float('inf')

        return {
            "time": self.time,
            "tip_position": tip_pos,
            "debris_position": debris_pos,
            "debris_velocity": self.debris.velocity.copy() if self.debris else np.zeros(3),
            "debris_orientation": self.debris.orientation.copy() if self.debris else np.array([1,0,0,0]),
            "distance": distance,
            "joint_angles": self.arm.joint_angles.copy(),
            "manipulability": self.arm.get_manipulability(),
            "gripper_open": self.arm.gripper_open,
        }
