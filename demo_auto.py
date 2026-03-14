"""
自动捕捉演示 - 展示AI驱动的太空垃圾自动追踪与抓取
使用运动预测 + MPC轨迹规划实现预判式抓取
"""
import numpy as np
import pygame
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env import SpaceEnvironment
from sim.sensors import VisionSensor
from prediction.lstm_predictor import PhysicsPredictor, MotionPredictor
from planning.mpc_planner import MPCPlanner
from planning.grasp_strategy import GraspStrategy
from visualization.renderer import SpaceRenderer


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AutoCaptureDemo:
    """自动捕捉演示"""

    def __init__(self):
        self.config = load_config()
        self.env = SpaceEnvironment(self.config)
        self.renderer = SpaceRenderer(
            width=self.config["display"]["width"],
            height=self.config["display"]["height"],
            star_count=self.config["display"]["star_count"]
        )

        # 使用物理预测器作为默认（无需训练）
        # 如果有训练好的LSTM模型，会自动加载
        self.predictor = self._load_predictor()
        self.planner = MPCPlanner(self.env.arm, self.config)
        self.strategy = GraspStrategy(self.config)

        # 状态
        self.state = "SEARCHING"  # SEARCHING -> TRACKING -> APPROACHING -> GRASPING -> DONE
        self.predicted_trajectory = None
        self.predicted_confidence = None

        # 参数面板
        self.debris_speed = 0.5
        self.debris_ang_speed = 0.1
        self.debris_shape = "satellite"
        self.shape_idx = 0
        self.shapes = ["satellite", "rocket_stage", "fragment"]

    def _load_predictor(self):
        """尝试加载LSTM模型，失败则使用物理预测器"""
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models", "lstm_predictor.pth")
        norm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "models", "normalization.npz")

        if os.path.exists(model_path) and os.path.exists(norm_path):
            try:
                pred_cfg = self.config["prediction"]
                predictor = MotionPredictor(
                    model_path=model_path,
                    hidden_size=pred_cfg["hidden_size"],
                    num_layers=pred_cfg["num_layers"],
                    prediction_horizon=pred_cfg["prediction_horizon"]
                )
                norm_data = np.load(norm_path)
                predictor.set_normalization(norm_data["mean"], norm_data["std"])
                print("Loaded LSTM predictor model")
                return predictor
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")

        print("Using physics-based predictor (train LSTM model for better results)")
        return PhysicsPredictor(
            prediction_horizon=self.config["prediction"]["prediction_horizon"],
            dt=self.config["simulation"]["dt"]
        )

    def reset(self):
        """重置仿真"""
        self.env.reset(
            linear_speed=self.debris_speed,
            angular_speed=self.debris_ang_speed,
            shape=self.debris_shape
        )
        self.planner.reset()
        self.renderer.clear_trails()
        self.state = "SEARCHING"
        self.predicted_trajectory = None
        self.predicted_confidence = None

    def run(self):
        """主循环"""
        self.reset()
        running = True
        paused = False
        show_prediction = True
        sim_speed = 1  # 仿真速度倍率

        while running:
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.env.done:
                            self.reset()
                        else:
                            paused = not paused
                    elif event.key == pygame.K_p:
                        show_prediction = not show_prediction
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_1:
                        self.debris_speed = max(0.1, self.debris_speed - 0.1)
                    elif event.key == pygame.K_2:
                        self.debris_speed = min(2.0, self.debris_speed + 0.1)
                    elif event.key == pygame.K_3:
                        self.debris_ang_speed = max(0.0, self.debris_ang_speed - 0.05)
                    elif event.key == pygame.K_4:
                        self.debris_ang_speed = min(0.524, self.debris_ang_speed + 0.05)
                    elif event.key == pygame.K_TAB:
                        self.shape_idx = (self.shape_idx + 1) % len(self.shapes)
                        self.debris_shape = self.shapes[self.shape_idx]
                    elif event.key == pygame.K_MINUS:
                        sim_speed = max(1, sim_speed - 1)
                    elif event.key == pygame.K_EQUALS:
                        sim_speed = min(10, sim_speed + 1)

            if paused or self.env.done:
                self._render(show_prediction, paused, sim_speed)
                continue

            # 仿真步进
            for _ in range(sim_speed):
                if self.env.done:
                    break
                self._simulation_step()

            # 相机更新
            keys = pygame.key.get_pressed()
            key_dict = {k: keys[k] for k in [pygame.K_LEFT, pygame.K_RIGHT,
                                               pygame.K_UP, pygame.K_DOWN,
                                               pygame.K_PAGEUP, pygame.K_PAGEDOWN]}
            self.renderer.update_camera(key_dict)

            # 自动跟踪相机
            if self.env.debris:
                mid = (self.env.arm.get_tip_position() + self.env.debris.position) / 2
                self.renderer.camera_target = self.renderer.camera_target * 0.95 + mid * 0.05

            self._render(show_prediction, paused, sim_speed)

        self.renderer.close()

    def _simulation_step(self):
        """单步仿真逻辑"""
        info = self.env._get_info()
        distance = info["distance"]

        # 获取传感器观测
        obs = self.env.sensor.observe(self.env.debris, self.env.time)
        obs_seq = self.env.sensor.get_observation_sequence(
            self.config["prediction"]["sequence_length"]
        )

        # 运动预测
        target_pos = None
        close_gripper = False

        if self.state == "SEARCHING":
            # 等待传感器积累足够数据
            if obs_seq is not None:
                self.state = "TRACKING"

        elif self.state == "TRACKING":
            # 预测目标轨迹
            if obs_seq is not None:
                self.predicted_trajectory, self.predicted_confidence = \
                    self.predictor.predict_position_only(obs_seq)

                # MPC规划
                target_pos = self.planner.plan(
                    self.predicted_trajectory,
                    confidence=self.predicted_confidence
                )

            if distance < 3.0:
                self.state = "APPROACHING"

        elif self.state == "APPROACHING":
            # 精确接近
            if obs_seq is not None:
                self.predicted_trajectory, self.predicted_confidence = \
                    self.predictor.predict_position_only(obs_seq)
                target_pos = self.planner.plan(
                    self.predicted_trajectory,
                    confidence=self.predicted_confidence
                )

            # 判断是否可以抓取
            tip_vel = self.env.arm.get_tip_velocity()
            debris_vel = self.env.debris.velocity
            rel_vel = np.linalg.norm(tip_vel - debris_vel)

            gripper_type = self.strategy.select_gripper(
                self.env.debris.shape, np.max(self.env.debris.size)
            )
            should_grasp, reason = self.strategy.should_grasp(
                distance, rel_vel, gripper_type
            )

            if should_grasp:
                self.state = "GRASPING"
                close_gripper = True

        elif self.state == "GRASPING":
            close_gripper = True
            # 继续跟踪
            if obs_seq is not None:
                self.predicted_trajectory, self.predicted_confidence = \
                    self.predictor.predict_position_only(obs_seq)
                target_pos = self.planner.plan(
                    self.predicted_trajectory,
                    confidence=self.predicted_confidence
                )

        # 执行仿真步
        info, done, success = self.env.step(target_pos, close_gripper)

        # 更新轨迹
        self.renderer.update_trail(
            self.env.debris.position,
            self.env.arm.get_tip_position()
        )

        if done:
            self.state = "DONE"

    def _render(self, show_prediction: bool, paused: bool, sim_speed: int):
        """渲染一帧"""
        self.renderer.draw_background()
        self.renderer.draw_grid()
        self.renderer.draw_axes(np.zeros(3), 1.0)

        # 绘制轨迹
        self.renderer.draw_trail(self.renderer.debris_trail, (255, 100, 100))
        self.renderer.draw_trail(self.renderer.arm_trail, (100, 200, 255))

        # 绘制预测轨迹
        if show_prediction and self.predicted_trajectory is not None:
            self.renderer.draw_predicted_trajectory(
                self.predicted_trajectory, self.predicted_confidence
            )

        # 绘制碎片
        if self.env.debris:
            self.renderer.draw_debris(
                self.env.debris.position,
                self.env.debris.orientation,
                self.env.debris.size,
                self.env.debris.shape
            )

        # 绘制机械臂
        joint_positions = self.env.arm.get_joint_positions()
        self.renderer.draw_robot_arm(
            joint_positions,
            self.env.arm.gripper_open,
            highlight=(self.state == "GRASPING")
        )

        # 绘制连接线
        if self.env.debris:
            self.renderer.draw_connection_line(
                self.env.arm.get_tip_position(),
                self.env.debris.position
            )

        # HUD
        info = self.env._get_info()
        info["state"] = self.state
        self.renderer.draw_hud(info, mode="AUTO")

        # 状态标签
        state_colors = {
            "SEARCHING": (150, 150, 150),
            "TRACKING": (255, 255, 60),
            "APPROACHING": (255, 160, 40),
            "GRASPING": (60, 255, 60),
            "DONE": (60, 255, 60) if self.env.success else (255, 60, 60),
        }
        state_text = self.renderer.font_large.render(
            f"State: {self.state}", True, state_colors.get(self.state, (255,255,255))
        )
        self.renderer.screen.blit(state_text, (10, 240))

        # 参数面板
        param_y = 280
        params = [
            f"Speed: {self.debris_speed:.1f} m/s [1/2]",
            f"Ang Speed: {self.debris_ang_speed:.2f} rad/s [3/4]",
            f"Shape: {self.debris_shape} [TAB]",
            f"Sim Speed: {sim_speed}x [-/+]",
        ]
        for text in params:
            surf = self.renderer.font_small.render(text, True, (180, 180, 180))
            self.renderer.screen.blit(surf, (10, param_y))
            param_y += 20

        # 操作说明
        instructions = [
            "SPACE: Pause/Restart | R: Reset | P: Toggle Prediction",
            "Arrow Keys: Rotate Camera | PgUp/PgDn: Zoom | ESC: Quit",
        ]
        self.renderer.draw_instructions(instructions)

        # 暂停提示
        if paused:
            pause_text = self.renderer.font_title.render("PAUSED", True, (255, 255, 60))
            rect = pause_text.get_rect(center=(self.renderer.width // 2, 50))
            self.renderer.screen.blit(pause_text, rect)

        # 结果
        if self.env.done:
            self.renderer.draw_result_overlay(self.env.success, self.env.time)

        self.renderer.flip(self.config["display"]["fps"])


if __name__ == "__main__":
    demo = AutoCaptureDemo()
    demo.run()
