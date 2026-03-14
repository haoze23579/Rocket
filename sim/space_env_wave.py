"""
波次太空环境 - 多碎片高速袭来，逐个抓取回收
碎片从远处高速飞来，机械臂需要：追踪→预判→拦截→抓取→拖回回收区
"""
import numpy as np
from typing import Optional, Dict, List, Tuple
from .debris import SpaceDebris
from .robot_arm import RobotArm
from .sensors import VisionSensor


class RecycleZone:
    """回收区"""
    def __init__(self, position: np.ndarray, radius: float = 1.5):
        self.position = position
        self.radius = radius
        self.collected_count = 0

    def check_deposit(self, obj_position: np.ndarray) -> bool:
        return np.linalg.norm(obj_position - self.position) < self.radius


class DebrisWave:
    """碎片波次管理"""
    def __init__(self):
        self.debris_list: List[SpaceDebris] = []
        self.active_idx = 0          # 当前活跃碎片索引
        self.states: List[str] = []  # "incoming" / "active" / "captured" / "recycled" / "missed"
        self.spawn_times: List[float] = []
        self.speeds: List[float] = []

    def add_debris(self, debris: SpaceDebris, spawn_time: float, speed: float):
        self.debris_list.append(debris)
        self.states.append("incoming")
        self.spawn_times.append(spawn_time)
        self.speeds.append(speed)

    @property
    def total(self) -> int:
        return len(self.debris_list)

    @property
    def recycled_count(self) -> int:
        return sum(1 for s in self.states if s == "recycled")

    @property
    def missed_count(self) -> int:
        return sum(1 for s in self.states if s == "missed")

    @property
    def all_done(self) -> bool:
        return all(s in ("recycled", "missed") for s in self.states)

    def get_current_debris(self) -> Optional[SpaceDebris]:
        """获取当前需要处理的碎片"""
        for i, s in enumerate(self.states):
            if s in ("incoming", "active", "captured"):
                return self.debris_list[i]
        return None

    def get_current_index(self) -> int:
        for i, s in enumerate(self.states):
            if s in ("incoming", "active", "captured"):
                return i
        return -1


class WaveSpaceEnvironment:
    """波次太空环境 - 多碎片高速袭来"""

    SHAPES = ["satellite", "rocket_stage", "fragment"]

    def __init__(self, config: dict):
        self.config = config
        self.dt = config["simulation"]["dt"]
        self.time = 0.0
        self.done = False

        # 机械臂
        arm_cfg = config["robot_arm"]
        self.arm = RobotArm(
            link_lengths=arm_cfg["link_lengths"],
            base_position=np.array(arm_cfg["base_position"]),
            joint_max_velocity=arm_cfg["joint_max_velocity"],
            joint_max_torque=arm_cfg["joint_max_torque"]
        )

        # 传感器
        vis_cfg = config["vision"]
        self.sensor = VisionSensor(
            frame_rate=vis_cfg["frame_rate"],
            noise_std=vis_cfg["noise_std"],
            max_range=vis_cfg["max_range"],
            fov=vis_cfg["fov"]
        )

        # 回收区 (机械臂基座后方)
        self.recycle_zone = RecycleZone(
            position=np.array([-3.0, 0.0, 0.0]),
            radius=1.5
        )

        # 波次
        self.wave: Optional[DebrisWave] = None
        self.current_debris: Optional[SpaceDebris] = None
        self.holding_debris = False  # 是否正在抓着碎片

        # 抓取参数
        self.grasp_dist_threshold = 0.3
        self.grasp_vel_threshold = 0.8  # 高速场景放宽速度阈值
        self.miss_distance = 25.0       # 碎片飞过这个距离算miss

        # 记录
        self.arm_tip_history: List[np.ndarray] = []
        self.events: List[Dict] = []  # 事件日志

    def reset(self, wave_config: Optional[Dict] = None) -> Dict:
        """
        重置环境，生成一波碎片
        wave_config: {"count": 5, "min_speed": 1.0, "max_speed": 4.0, "interval": 3.0}
        """
        self.time = 0.0
        self.done = False
        self.holding_debris = False
        self.arm_tip_history.clear()
        self.events.clear()
        self.sensor.reset()

        if wave_config is None:
            wave_config = {"count": 5, "min_speed": 1.5, "max_speed": 5.0, "interval": 4.0}

        # 重置机械臂
        init_angles = np.array([0.0, -0.3, 0.6, 0.0, -0.3, 0.0])
        self.arm.reset(init_angles)

        # 生成波次
        self.wave = DebrisWave()
        self.recycle_zone.collected_count = 0

        for i in range(wave_config["count"]):
            speed = np.random.uniform(wave_config["min_speed"], wave_config["max_speed"])
            spawn_time = i * wave_config["interval"] + np.random.uniform(0, 1.0)

            # 碎片从随机方向高速飞来
            # 主要从前方(+X)飞来，带一些随机偏移
            approach_dir = np.array([-1.0, 0.0, 0.0])  # 主方向
            approach_dir += np.random.randn(3) * 0.4     # 随机偏移
            approach_dir /= np.linalg.norm(approach_dir)

            spawn_dist = 18.0 + np.random.uniform(0, 5.0)
            init_pos = -approach_dir * spawn_dist  # 从远处出发

            velocity = approach_dir * speed

            # 随机角速度（翻滚）
            ang_speed = np.random.uniform(0.1, 0.8)
            ang_dir = np.random.randn(3)
            ang_dir /= np.linalg.norm(ang_dir)
            ang_vel = ang_dir * ang_speed

            shape = np.random.choice(self.SHAPES)
            debris = SpaceDebris(
                shape=shape,
                position=init_pos,
                velocity=velocity,
                angular_velocity=ang_vel
            )

            self.wave.add_debris(debris, spawn_time, speed)

        self.current_debris = None
        self._activate_next_debris()

        return self._get_info()

    def _activate_next_debris(self):
        """激活下一个碎片"""
        if self.wave is None:
            return
        idx = self.wave.get_current_index()
        if idx >= 0 and self.wave.states[idx] == "incoming":
            self.wave.states[idx] = "active"
            self.current_debris = self.wave.debris_list[idx]
            self.sensor.reset()
            self.holding_debris = False
            self.arm.open_gripper()
            self.events.append({
                "time": self.time,
                "type": "debris_activated",
                "index": idx,
                "speed": self.wave.speeds[idx],
                "shape": self.current_debris.shape
            })

    def step(self, target_pos: Optional[np.ndarray] = None,
             close_gripper: bool = False) -> Tuple[Dict, bool]:
        """
        仿真前进一步
        返回: (info, wave_all_done)
        """
        # 检查是否有新碎片需要激活
        if self.wave and self.current_debris is None:
            for i, (state, spawn_t) in enumerate(zip(self.wave.states, self.wave.spawn_times)):
                if state == "incoming" and self.time >= spawn_t:
                    self.wave.states[i] = "active"
                    self.current_debris = self.wave.debris_list[i]
                    self.sensor.reset()
                    self.holding_debris = False
                    self.arm.open_gripper()
                    self.events.append({
                        "time": self.time, "type": "debris_activated",
                        "index": i, "speed": self.wave.speeds[i]
                    })
                    break

        # 更新所有活跃碎片的物理
        if self.wave:
            for i, debris in enumerate(self.wave.debris_list):
                if self.wave.states[i] == "active":
                    if self.holding_debris and debris is self.current_debris:
                        # 被抓住的碎片跟随机械臂末端
                        tip = self.arm.get_tip_position()
                        debris.position = tip.copy()
                        debris.velocity = self.arm.get_tip_velocity()
                    else:
                        debris.step(self.dt)

        # 机械臂运动
        if target_pos is not None:
            self.arm.move_to_target(target_pos, self.dt)

        # 夹爪控制
        if close_gripper:
            self.arm.close_gripper()
        else:
            if self.holding_debris:
                # 松开夹爪 = 释放碎片
                pass  # 保持抓取状态直到回收
            # 不主动打开，由回收逻辑控制

        # 传感器观测当前碎片
        if self.current_debris and not self.holding_debris:
            self.sensor.observe(self.current_debris, self.time)

        # 抓取检测
        if self.current_debris and close_gripper and not self.holding_debris:
            tip = self.arm.get_tip_position()
            dist = np.linalg.norm(tip - self.current_debris.position)
            tip_vel = self.arm.get_tip_velocity()
            rel_vel = np.linalg.norm(tip_vel - self.current_debris.velocity)

            if dist < self.grasp_dist_threshold and rel_vel < self.grasp_vel_threshold:
                self.holding_debris = True
                self.events.append({
                    "time": self.time, "type": "captured",
                    "index": self.wave.get_current_index(),
                    "distance": dist, "rel_velocity": rel_vel
                })
                idx = self.wave.get_current_index()
                if idx >= 0:
                    self.wave.states[idx] = "captured"

        # 回收检测
        if self.holding_debris and self.current_debris:
            tip = self.arm.get_tip_position()
            if self.recycle_zone.check_deposit(tip):
                idx = self.wave.get_current_index()
                if idx >= 0:
                    self.wave.states[idx] = "recycled"
                    self.recycle_zone.collected_count += 1
                    self.events.append({
                        "time": self.time, "type": "recycled",
                        "index": idx
                    })
                self.holding_debris = False
                self.current_debris = None
                self.arm.open_gripper()
                # 激活下一个
                self._activate_next_debris()

        # 碎片飞过检测（miss）
        if self.current_debris and not self.holding_debris:
            idx = self.wave.get_current_index()
            if idx >= 0 and self.wave.states[idx] == "active":
                # 碎片飞过机械臂基座太远
                debris_pos = self.current_debris.position
                if np.linalg.norm(debris_pos) > self.miss_distance:
                    self.wave.states[idx] = "missed"
                    self.events.append({
                        "time": self.time, "type": "missed", "index": idx
                    })
                    self.current_debris = None
                    self._activate_next_debris()

        # 记录
        self.arm_tip_history.append(self.arm.get_tip_position().copy())

        self.time += self.dt

        # 检查波次是否全部完成
        wave_done = self.wave.all_done if self.wave else True
        if wave_done:
            self.done = True

        return self._get_info(), wave_done

    def _get_info(self) -> Dict:
        tip_pos = self.arm.get_tip_position()
        debris_pos = self.current_debris.position if self.current_debris else np.zeros(3)
        debris_vel = self.current_debris.velocity if self.current_debris else np.zeros(3)
        distance = np.linalg.norm(tip_pos - debris_pos) if self.current_debris else float('inf')

        return {
            "time": self.time,
            "tip_position": tip_pos,
            "debris_position": debris_pos,
            "debris_velocity": debris_vel,
            "debris_orientation": self.current_debris.orientation.copy() if self.current_debris else np.array([1,0,0,0]),
            "distance": distance,
            "joint_angles": self.arm.joint_angles.copy(),
            "manipulability": self.arm.get_manipulability(),
            "gripper_open": self.arm.gripper_open,
            "holding": self.holding_debris,
            "wave_total": self.wave.total if self.wave else 0,
            "wave_recycled": self.wave.recycled_count if self.wave else 0,
            "wave_missed": self.wave.missed_count if self.wave else 0,
            "current_debris_shape": self.current_debris.shape if self.current_debris else "",
            "current_debris_speed": np.linalg.norm(debris_vel),
            "recycle_zone_pos": self.recycle_zone.position.copy(),
        }

    def get_all_debris_positions(self) -> List[Tuple[np.ndarray, str, str, np.ndarray, np.ndarray, float]]:
        """获取所有碎片的位置信息，用于渲染
        返回: [(position, shape, state, orientation, velocity, size_scale), ...]
        """
        result = []
        if self.wave is None:
            return result
        for i, (debris, state) in enumerate(zip(self.wave.debris_list, self.wave.states)):
            if state in ("active", "captured"):
                speed = np.linalg.norm(debris.velocity)
                result.append((
                    debris.position.copy(),
                    debris.shape,
                    state,
                    debris.orientation.copy(),
                    debris.velocity.copy(),
                    speed
                ))
            elif state == "incoming":
                # 还没出现的碎片，检查是否快到了
                if self.wave.spawn_times[i] - self.time < 2.0:
                    # 预警：即将到来
                    result.append((
                        debris.position.copy(),
                        debris.shape,
                        "warning",
                        debris.orientation.copy(),
                        debris.velocity.copy(),
                        np.linalg.norm(debris.velocity)
                    ))
        return result
