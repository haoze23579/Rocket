"""
3D手动捕捉游戏 - 波次模式
多个太空碎片高速袭来，玩家手动操控机械臂逐个拦截、抓取、拖回回收区
关卡递增：碎片越来越快、越来越多
"""
import numpy as np
import pygame
import yaml
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env_wave import WaveSpaceEnvironment
from prediction.lstm_predictor import PhysicsPredictor
from visualization.renderer_3d import SpaceRenderer3D


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ManualCaptureGame3D:

    def __init__(self):
        self.config = load_config()
        self.env = WaveSpaceEnvironment(self.config)
        self.renderer = SpaceRenderer3D(
            width=self.config["display"]["width"],
            height=self.config["display"]["height"],
            star_count=2000
        )
        self.predictor = PhysicsPredictor(
            prediction_horizon=self.config["prediction"]["prediction_horizon"],
            dt=self.config["simulation"]["dt"]
        )

        # 游戏状态
        self.score = 0
        self.level = 1
        self.total_recycled = 0
        self.total_missed = 0
        self.show_assist = True
        self.manual_target = np.array([2.0, 1.0, 0.0])
        self.move_speed = 0.10

        # 鼠标
        self.mouse_dragging = False
        self.last_mouse_pos = (0, 0)

        # 关卡设计：碎片数量递增、速度递增
        self.level_configs = [
            {"count": 2, "min_speed": 0.8, "max_speed": 1.5, "interval": 6.0},
            {"count": 3, "min_speed": 1.0, "max_speed": 2.0, "interval": 5.5},
            {"count": 3, "min_speed": 1.5, "max_speed": 2.5, "interval": 5.0},
            {"count": 4, "min_speed": 1.5, "max_speed": 3.0, "interval": 4.5},
            {"count": 4, "min_speed": 2.0, "max_speed": 3.5, "interval": 4.0},
            {"count": 5, "min_speed": 2.0, "max_speed": 4.0, "interval": 4.0},
            {"count": 5, "min_speed": 2.5, "max_speed": 4.5, "interval": 3.5},
            {"count": 6, "min_speed": 3.0, "max_speed": 5.0, "interval": 3.5},
            {"count": 7, "min_speed": 3.0, "max_speed": 5.5, "interval": 3.0},
            {"count": 8, "min_speed": 3.5, "max_speed": 6.0, "interval": 3.0},
        ]

    def get_wave_config(self) -> dict:
        idx = min(self.level - 1, len(self.level_configs) - 1)
        return self.level_configs[idx].copy()

    def reset_wave(self):
        cfg = self.get_wave_config()
        self.env.reset(cfg)
        self.renderer.clear_trails()
        self.manual_target = np.array([2.0, 1.0, 0.0])

    def run(self):
        self.reset_wave()
        running = True
        gripper_pressed = False

        self._show_title_screen()

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
                            # 结算
                            recycled = self.env.wave.recycled_count
                            missed = self.env.wave.missed_count
                            self.total_recycled += recycled
                            self.total_missed += missed
                            self.score += self._calculate_wave_score(recycled, missed)
                            # 升级条件：回收超过一半
                            if recycled > self.env.wave.total // 2:
                                self.level = min(self.level + 1, len(self.level_configs))
                            self.reset_wave()
                    elif event.key == pygame.K_h:
                        self.show_assist = not self.show_assist
                    elif event.key == pygame.K_r:
                        self.reset_wave()
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

            if self.env.done:
                self._render_game(gripper_pressed)
                continue

            # 手动控制
            keys = pygame.key.get_pressed()
            gripper_pressed = keys[pygame.K_g]

            # WASD控制 (相对相机方向)
            if keys[pygame.K_w]:
                self.manual_target[2] -= self.move_speed
            if keys[pygame.K_s]:
                self.manual_target[2] += self.move_speed
            if keys[pygame.K_a]:
                self.manual_target[0] -= self.move_speed
            if keys[pygame.K_d]:
                self.manual_target[0] += self.move_speed
            if keys[pygame.K_q]:
                self.manual_target[1] += self.move_speed
            if keys[pygame.K_e]:
                self.manual_target[1] -= self.move_speed

            # Shift加速
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                extra = 0.08
                if keys[pygame.K_w]:
                    self.manual_target[2] -= extra
                if keys[pygame.K_s]:
                    self.manual_target[2] += extra
                if keys[pygame.K_a]:
                    self.manual_target[0] -= extra
                if keys[pygame.K_d]:
                    self.manual_target[0] += extra
                if keys[pygame.K_q]:
                    self.manual_target[1] += extra
                if keys[pygame.K_e]:
                    self.manual_target[1] -= extra

            # 限制臂展
            arm_reach = sum(self.env.arm.link_lengths)
            dist = np.linalg.norm(self.manual_target - self.env.arm.base_position)
            if dist > arm_reach * 0.95:
                direction = self.manual_target - self.env.arm.base_position
                self.manual_target = self.env.arm.base_position + direction / dist * arm_reach * 0.95

            # F键追踪碎片
            if keys[pygame.K_f] and self.env.current_debris and not self.env.holding_debris:
                self.manual_target = self.env.current_debris.position.copy()

            # V键快速回到回收区
            if keys[pygame.K_v] and self.env.holding_debris:
                rz = self.env.recycle_zone.position.copy()
                rz[1] = 0.5
                self.manual_target = rz

            # 仿真步进
            info, wave_done = self.env.step(self.manual_target, gripper_pressed)

            # 更新轨迹
            if self.env.current_debris:
                self.renderer.update_trail(
                    self.env.current_debris.position,
                    self.env.arm.get_tip_position()
                )

            # 推进器粒子
            tip = self.env.arm.get_tip_position()
            vel = self.env.arm.get_tip_velocity()
            if np.linalg.norm(vel) > 0.1:
                self.renderer.thruster_particles.emit(
                    tip, -vel * 0.3,
                    color=(1.0, 0.6, 0.1, 0.8), count=2, spread=0.15
                )
            self.renderer.thruster_particles.update(self.config["simulation"]["dt"])

            # 事件特效
            for evt in self.env.events:
                if evt["time"] >= self.env.time - self.config["simulation"]["dt"]:
                    if evt["type"] == "recycled":
                        self.renderer.spark_particles.emit(
                            tip, np.zeros(3),
                            color=(0.2, 1.0, 0.4, 1.0), count=60, spread=1.5, lifetime=2.0
                        )
                        self.renderer.clear_trails()
                    elif evt["type"] == "captured":
                        self.renderer.spark_particles.emit(
                            tip, np.zeros(3),
                            color=(1.0, 1.0, 0.3, 1.0), count=30, spread=0.8, lifetime=1.0
                        )

            # 键盘相机
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

            self._render_game(gripper_pressed)

        self.renderer.close()

    def _show_title_screen(self):
        waiting = True
        angle = 0
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.renderer.close()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False

            self.renderer.begin_frame()
            self.renderer.cam_yaw = angle
            self.renderer.cam_pitch = 20
            self.renderer.cam_distance = 15
            angle += 0.3

            self.renderer.draw_stars()
            self.renderer.draw_grid()

            # 展示旋转碎片
            t = angle * 0.02
            orient = np.array([math.cos(t), math.sin(t), 0, 0])
            orient /= np.linalg.norm(orient)
            self.renderer.draw_debris(
                np.array([3.0, 1.0, 0.0]), orient,
                np.array([2.0, 1.5, 1.0]), "satellite"
            )
            self.renderer.draw_debris(
                np.array([-2.0, 2.0, 3.0]), orient,
                np.array([4.0, 1.2, 1.2]), "rocket_stage"
            )

            # 回收区
            self.renderer.draw_recycle_zone(
                np.array([-3.0, 0.0, 0.0]), 1.5, 0, angle * 0.05
            )

            self.renderer.begin_2d_overlay()
            cx = self.renderer.width // 2
            self.renderer._draw_text_2d(
                cx - 220, 100, "SPACE DEBRIS CAPTURE",
                self.renderer.hud_font_title, (60, 220, 255)
            )
            self.renderer._draw_text_2d(
                cx - 160, 150, "High-Speed Intercept Mode",
                self.renderer.hud_font_large, (255, 200, 100)
            )

            controls = [
                "WASD: Move arm (XZ) | Q/E: Up/Down | Shift: Boost",
                "G (hold): Close gripper to grab debris",
                "F: Auto-track current debris",
                "V: Quick-return to recycle zone (while holding)",
                "H: Toggle prediction assist line",
                "Mouse Drag: Rotate camera | Scroll: Zoom",
                "SPACE: Next wave | R: Reset | ESC: Quit",
                "",
                "GOAL: Grab each debris and drag it to the",
                "green recycle zone before it flies past!",
            ]
            y = 210
            for text in controls:
                if text:
                    self.renderer._draw_text_2d(cx - 220, y, text,
                                                self.renderer.hud_font_small, (170, 170, 170))
                y += 24

            self.renderer._draw_text_2d(
                cx - 130, y + 20, "Press any key to start",
                self.renderer.hud_font_large, (255, 255, 100)
            )
            self.renderer.end_2d_overlay()

            self.renderer.end_frame(30)

    def _render_game(self, gripper_pressed: bool):
        self.renderer.begin_frame()

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

        # 所有碎片 + 高速拖尾
        all_debris = self.env.get_all_debris_positions()
        for pos, shape, dstate, orient, vel, speed in all_debris:
            if dstate == "warning":
                self.renderer.draw_warning_indicator(pos, speed)
            else:
                trail_color = (1.0, 0.3, 0.1) if speed > 3.0 else (1.0, 0.6, 0.2)
                self.renderer.draw_speed_trail(pos, vel, trail_color)
                size = np.array([2.0, 1.5, 1.0])
                self.renderer.draw_debris(pos, orient, size, shape)

        # 轨迹
        self.renderer.draw_trail(self.renderer.debris_trail, (1.0, 0.4, 0.4))
        self.renderer.draw_trail(self.renderer.arm_trail, (0.4, 0.7, 1.0))

        # 辅助预测线
        if self.show_assist and self.env.current_debris and not self.env.holding_debris:
            obs_seq = self.env.sensor.get_observation_sequence(
                self.config["prediction"]["sequence_length"]
            )
            if obs_seq is not None:
                pred_pos, pred_conf = self.predictor.predict_position_only(obs_seq)
                self.renderer.draw_predicted_trajectory(pred_pos, pred_conf)

        # 机械臂
        joint_positions = self.env.arm.get_joint_positions()
        self.renderer.draw_robot_arm(joint_positions, not gripper_pressed,
                                     highlight=gripper_pressed)

        # 目标标记
        self.renderer.draw_target_marker(self.manual_target)

        # 连接线
        if self.env.current_debris and not self.env.holding_debris:
            self.renderer.draw_connection_line(
                self.env.arm.get_tip_position(),
                self.env.current_debris.position
            )
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
        self.renderer.draw_wave_hud(info, mode="MANUAL")

        # 游戏面板
        self.renderer.begin_2d_overlay()
        px = self.renderer.width - 260
        py = 280
        self.renderer._draw_panel_bg(px, py, 250, 180,
                                      border_color=(1.0, 0.85, 0.2, 0.8))
        y = py + 10
        cfg = self.get_wave_config()
        lines = [
            (f"LEVEL: {self.level}", (255, 215, 0)),
            (f"SCORE: {self.score}", (255, 255, 255)),
            (f"Total Recycled: {self.total_recycled}", (100, 255, 100)),
            (f"Total Missed: {self.total_missed}", (255, 100, 100)),
            (f"Wave: {cfg['count']} debris", (100, 220, 255)),
            (f"Speed: {cfg['min_speed']:.1f}-{cfg['max_speed']:.1f} m/s", (255, 180, 100)),
            (f"Assist: {'ON' if self.show_assist else 'OFF'} [H]", (255, 255, 100)),
        ]
        for text, color in lines:
            self.renderer._draw_text_2d(px + 10, y, text,
                                        self.renderer.hud_font_small, color)
            y += 22
        self.renderer.end_2d_overlay()

        # 操作说明
        self.renderer.draw_instructions([
            "WASD+QE: Move | Shift: Boost | G: Grab | F: Track | V: Return | Mouse: Camera",
            "SPACE: Next Wave | R: Reset | H: Assist | ESC: Quit",
        ])

        # 波次完成
        if self.env.done:
            self._draw_wave_result()

        self.renderer.end_frame(self.config["display"]["fps"])

    def _calculate_wave_score(self, recycled: int, missed: int) -> int:
        cfg = self.get_wave_config()
        base = recycled * 150
        speed_bonus = int(cfg["max_speed"] * 50) * recycled
        time_bonus = max(0, int((60 - self.env.time) * 5))
        perfect_bonus = 500 if missed == 0 and recycled == cfg["count"] else 0
        return base + speed_bonus + time_bonus + perfect_bonus

    def _draw_wave_result(self):
        self.renderer.begin_2d_overlay()

        from OpenGL.GL import glColor4f, glBegin, glEnd, glVertex2f, GL_QUADS
        glColor4f(0, 0, 0, 0.65)
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
        wave_score = self._calculate_wave_score(recycled, missed)

        title_color = (100, 255, 100) if recycled > total // 2 else (255, 200, 100)
        self.renderer._draw_text_2d(cx - 140, cy - 100, "WAVE COMPLETE",
                                     self.renderer.hud_font_title, title_color)
        self.renderer._draw_text_2d(cx - 130, cy - 50,
                                     f"Recycled: {recycled}/{total} ({rate:.0f}%)",
                                     self.renderer.hud_font_large, (100, 255, 100))
        self.renderer._draw_text_2d(cx - 80, cy - 15,
                                     f"Missed: {missed}",
                                     self.renderer.hud_font_large, (255, 100, 100))
        self.renderer._draw_text_2d(cx - 100, cy + 20,
                                     f"Wave Score: +{wave_score}",
                                     self.renderer.hud_font_large, (255, 215, 0))
        self.renderer._draw_text_2d(cx - 80, cy + 55,
                                     f"Time: {self.env.time:.1f}s",
                                     self.renderer.hud_font_large, (220, 220, 220))

        if recycled == total and missed == 0:
            self.renderer._draw_text_2d(cx - 100, cy + 90,
                                         "PERFECT WAVE! +500",
                                         self.renderer.hud_font_large, (255, 215, 0))

        next_text = "LEVEL UP! " if recycled > total // 2 else ""
        self.renderer._draw_text_2d(cx - 180, cy + 130,
                                     f"{next_text}Press SPACE for next wave, ESC to quit",
                                     self.renderer.hud_font_small, (150, 150, 150))

        self.renderer.end_2d_overlay()


if __name__ == "__main__":
    game = ManualCaptureGame3D()
    game.run()
