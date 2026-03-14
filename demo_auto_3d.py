"""
3D自动捕捉演示 - 波次模式
多个太空碎片高速袭来，AI自动逐个追踪→预判→拦截→抓取→拖回回收区
"""
import numpy as np
import pygame
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env_wave import WaveSpaceEnvironment
from prediction.lstm_predictor import PhysicsPredictor, MotionPredictor
from planning.mpc_planner import MPCPlanner
from planning.grasp_strategy import GraspStrategy
from visualization.renderer_3d import SpaceRenderer3D


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class AutoCaptureDemo3D:

    def __init__(self):
        self.config = load_config()
        self.env = WaveSpaceEnvironment(self.config)
        self.renderer = SpaceRenderer3D(
            width=self.config["display"]["width"],
            height=self.config["display"]["height"],
            star_count=2000
        )
        self.predictor = self._load_predictor()
        self.planner = MPCPlanner(self.env.arm, self.config)
        self.strategy = GraspStrategy(self.config)

        self.state = "WAITING"  # WAITING/TRACKING/APPROACHING/GRASPING/RETURNING/DONE
        self.predicted_trajectory = None
        self.predicted_confidence = None

        # 波次参数
        self.wave_config = {
            "count": 5, "min_speed": 1.5, "max_speed": 4.0, "interval": 5.0
        }

        # 鼠标
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)

    def _load_predictor(self):
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
                print("[INFO] Loaded LSTM predictor")
                return predictor
            except Exception as e:
                print(f"[WARN] LSTM load failed: {e}")
        print("[INFO] Using physics predictor")
        return PhysicsPredictor(
            prediction_horizon=self.config["prediction"]["prediction_horizon"],
            dt=self.config["simulation"]["dt"]
        )

    def reset(self):
        self.env.reset(self.wave_config)
        self.planner.reset()
        self.renderer.clear_trails()
        self.state = "WAITING"
        self.predicted_trajectory = None
        self.predicted_confidence = None

    def run(self):
        self.reset()
        running = True
        paused = False
        show_prediction = True
        sim_speed = 1

        while running:
            scroll = 0
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
                        self.wave_config["min_speed"] = max(0.5, self.wave_config["min_speed"] - 0.5)
                    elif event.key == pygame.K_2:
                        self.wave_config["min_speed"] = min(5.0, self.wave_config["min_speed"] + 0.5)
                    elif event.key == pygame.K_3:
                        self.wave_config["count"] = max(1, self.wave_config["count"] - 1)
                    elif event.key == pygame.K_4:
                        self.wave_config["count"] = min(15, self.wave_config["count"] + 1)
                    elif event.key == pygame.K_MINUS:
                        sim_speed = max(1, sim_speed - 1)
                    elif event.key == pygame.K_EQUALS:
                        sim_speed = min(10, sim_speed + 1)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_dragging = True
                        self.last_mouse_pos = event.pos
                    elif event.button == 4:
                        scroll = 1
                    elif event.button == 5:
                        scroll = -1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_dragging:
                        dx = event.pos[0] - self.last_mouse_pos[0]
                        dy = event.pos[1] - self.last_mouse_pos[1]
                        self.renderer.update_camera_mouse(dx, dy)
                        self.last_mouse_pos = event.pos

            if scroll != 0:
                self.renderer.update_camera_mouse(0, 0, scroll)

            if not paused and not self.env.done:
                for _ in range(sim_speed):
                    if self.env.done:
                        break
                    self._simulation_step()

            # 键盘相机
            keys = pygame.key.get_pressed()
            key_dict = {k: keys[k] for k in [pygame.K_LEFT, pygame.K_RIGHT,
                                               pygame.K_UP, pygame.K_DOWN,
                                               pygame.K_PAGEUP, pygame.K_PAGEDOWN]}
            self.renderer.update_camera(key_dict)

            # 相机跟踪
            if self.env.current_debris:
                mid = (self.env.arm.get_tip_position() + self.env.current_debris.position) / 2
                self.renderer.cam_target = mid
            else:
                self.renderer.cam_target = self.env.arm.get_tip_position()

            self._render(show_prediction, paused, sim_speed)

        self.renderer.close()

    def _simulation_step(self):
        info = self.env._get_info()
        distance = info["distance"]
        has_debris = self.env.current_debris is not None

        target_pos = None
        close_gripper = False

        if not has_debris:
            self.state = "WAITING"
            # 回到待机位置
            target_pos = np.array([2.0, 1.0, 0.0])
            self.predicted_trajectory = None
        elif self.env.holding_debris:
            # 正在抓着碎片，拖回回收区
            self.state = "RETURNING"
            target_pos = self.env.recycle_zone.position.copy()
            target_pos[1] = 0.5  # 稍微抬高
            close_gripper = True
            self.predicted_trajectory = None

            # 回收区粒子
            rz = self.env.recycle_zone.position
            dist_to_rz = np.linalg.norm(self.env.arm.get_tip_position() - rz)
            if dist_to_rz < 2.0:
                self.renderer.thruster_particles.emit(
                    self.env.arm.get_tip_position(), np.array([0, -0.5, 0]),
                    color=(0.2, 1.0, 0.4, 0.8), count=2, spread=0.3
                )
        else:
            # 追踪/接近碎片
            obs = self.env.sensor.observe(self.env.current_debris, self.env.time)
            obs_seq = self.env.sensor.get_observation_sequence(
                self.config["prediction"]["sequence_length"]
            )

            if obs_seq is None:
                self.state = "WAITING"
            else:
                self.predicted_trajectory, self.predicted_confidence = \
                    self.predictor.predict_position_only(obs_seq)
                target_pos = self.planner.plan(
                    self.predicted_trajectory, confidence=self.predicted_confidence
                )

                if distance < 3.0:
                    self.state = "APPROACHING"
                else:
                    self.state = "TRACKING"

                # 抓取判断
                if distance < self.env.grasp_dist_threshold * 2:
                    tip_vel = self.env.arm.get_tip_velocity()
                    debris_vel = self.env.current_debris.velocity
                    rel_vel = np.linalg.norm(tip_vel - debris_vel)
                    gripper_type = self.strategy.select_gripper(
                        self.env.current_debris.shape,
                        np.max(self.env.current_debris.size)
                    )
                    if distance < self.env.grasp_dist_threshold and rel_vel < self.env.grasp_vel_threshold:
                        self.state = "GRASPING"
                        close_gripper = True

        # 执行仿真步
        info, wave_done = self.env.step(target_pos, close_gripper)

        # 更新轨迹
        if self.env.current_debris:
            self.renderer.update_trail(
                self.env.current_debris.position,
                self.env.arm.get_tip_position()
            )

        # 推进器粒子
        if target_pos is not None:
            tip = self.env.arm.get_tip_position()
            vel = self.env.arm.get_tip_velocity()
            if np.linalg.norm(vel) > 0.1:
                self.renderer.thruster_particles.emit(
                    tip, -vel * 0.3,
                    color=(1.0, 0.6, 0.1, 0.8), count=2, spread=0.15
                )

        self.renderer.thruster_particles.update(self.config["simulation"]["dt"])

        # 检查事件
        for evt in self.env.events:
            if evt["time"] >= self.env.time - self.config["simulation"]["dt"]:
                if evt["type"] == "recycled":
                    tip = self.env.arm.get_tip_position()
                    self.renderer.spark_particles.emit(
                        tip, np.zeros(3),
                        color=(0.2, 1.0, 0.4, 1.0), count=60, spread=1.5, lifetime=2.0
                    )
                    self.renderer.clear_trails()
                elif evt["type"] == "captured":
                    tip = self.env.arm.get_tip_position()
                    self.renderer.spark_particles.emit(
                        tip, np.zeros(3),
                        color=(1.0, 1.0, 0.3, 1.0), count=30, spread=0.8, lifetime=1.0
                    )

        if wave_done:
            self.state = "DONE"

    def _render(self, show_prediction, paused, sim_speed):
        self.renderer.begin_frame()

        # 3D场景
        self.renderer.draw_stars()
        self.renderer.draw_grid()
        self.renderer.draw_axes()

        # 回收区
        self.renderer.draw_recycle_zone(
            self.env.recycle_zone.position,
            self.env.recycle_zone.radius,
            self.env.recycle_zone.collected_count,
            pulse_time=self.env.time
        )

        # 所有碎片（含高速拖尾）
        all_debris = self.env.get_all_debris_positions()
        for pos, shape, dstate, orient, vel, speed in all_debris:
            if dstate == "warning":
                self.renderer.draw_warning_indicator(pos, speed)
            else:
                # 高速拖尾
                trail_color = (1.0, 0.4, 0.2) if speed > 3.0 else (1.0, 0.7, 0.3)
                self.renderer.draw_speed_trail(pos, vel, trail_color)
                # 碎片本体
                size = np.array([2.0, 1.5, 1.0])  # 默认尺寸
                self.renderer.draw_debris(pos, orient, size, shape)

        # 轨迹
        self.renderer.draw_trail(self.renderer.debris_trail, (1.0, 0.4, 0.4))
        self.renderer.draw_trail(self.renderer.arm_trail, (0.4, 0.7, 1.0))

        # 预测轨迹
        if show_prediction and self.predicted_trajectory is not None:
            self.renderer.draw_predicted_trajectory(
                self.predicted_trajectory, self.predicted_confidence
            )

        # 机械臂
        joint_positions = self.env.arm.get_joint_positions()
        self.renderer.draw_robot_arm(
            joint_positions, self.env.arm.gripper_open,
            highlight=(self.state in ("GRASPING", "RETURNING"))
        )

        # 连接线
        if self.env.current_debris and not self.env.holding_debris:
            self.renderer.draw_connection_line(
                self.env.arm.get_tip_position(),
                self.env.current_debris.position
            )

        # 回收区方向指示（抓住碎片后）
        if self.env.holding_debris:
            self.renderer.draw_connection_line(
                self.env.arm.get_tip_position(),
                self.env.recycle_zone.position
            )

        # 粒子
        self.renderer.thruster_particles.draw()
        self.renderer.spark_particles.draw()
        self.renderer.spark_particles.update(0.016)

        # HUD
        info = self.env._get_info()
        self.renderer.draw_wave_hud(info, mode="AUTO")

        # 状态标签
        self.renderer.draw_state_label(self.state)

        # 参数面板
        self.renderer.begin_2d_overlay()
        y = 280
        params = [
            f"Min Speed: {self.wave_config['min_speed']:.1f} m/s [1/2]",
            f"Max Speed: {self.wave_config['max_speed']:.1f} m/s",
            f"Wave Size: {self.wave_config['count']} [3/4]",
            f"Sim Speed: {sim_speed}x [-/+]",
        ]
        for text in params:
            self.renderer._draw_text_2d(10, y, text, self.renderer.hud_font_small, (180, 180, 180))
            y += 20
        self.renderer.end_2d_overlay()

        # 操作说明
        self.renderer.draw_instructions([
            "SPACE: Pause/Restart | R: Reset | P: Prediction | Mouse: Camera",
            "1/2: Speed | 3/4: Wave Size | -/+: Sim Speed | ESC: Quit",
        ])

        if paused and not self.env.done:
            self.renderer.begin_2d_overlay()
            self.renderer._draw_text_2d(
                self.renderer.width // 2 - 60, 50,
                "PAUSED", self.renderer.hud_font_title, (255, 255, 60)
            )
            self.renderer.end_2d_overlay()

        # 波次完成结果
        if self.env.done:
            self._draw_wave_result()

        self.renderer.end_frame(self.config["display"]["fps"])

    def _draw_wave_result(self):
        self.renderer.begin_2d_overlay()

        # 半透明背景
        from OpenGL.GL import glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS
        glColor4f(0, 0, 0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(0, 0)
        glVertex2f(self.renderer.width, 0)
        glVertex2f(self.renderer.width, self.renderer.height)
        glVertex2f(0, self.renderer.height)
        glEnd()

        cx = self.renderer.width // 2
        cy = self.renderer.height // 2

        total = self.env.wave.total
        recycled = self.env.wave.recycled_count
        missed = self.env.wave.missed_count
        rate = recycled / max(1, total) * 100

        self.renderer._draw_text_2d(cx - 140, cy - 80, "WAVE COMPLETE",
                                     self.renderer.hud_font_title, (60, 220, 255))
        self.renderer._draw_text_2d(cx - 120, cy - 30,
                                     f"Recycled: {recycled}/{total} ({rate:.0f}%)",
                                     self.renderer.hud_font_large, (100, 255, 100))
        self.renderer._draw_text_2d(cx - 80, cy + 10,
                                     f"Missed: {missed}",
                                     self.renderer.hud_font_large, (255, 100, 100))
        self.renderer._draw_text_2d(cx - 80, cy + 50,
                                     f"Time: {self.env.time:.1f}s",
                                     self.renderer.hud_font_large, (220, 220, 220))
        self.renderer._draw_text_2d(cx - 160, cy + 100,
                                     "Press SPACE to restart, ESC to quit",
                                     self.renderer.hud_font_small, (150, 150, 150))

        self.renderer.end_2d_overlay()


if __name__ == "__main__":
    demo = AutoCaptureDemo3D()
    demo.run()
