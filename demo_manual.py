"""
手动捕捉演示 - 类似游戏的交互式太空垃圾捕获仿真
玩家通过键盘/鼠标控制机械臂末端，手动追踪并抓取太空碎片
"""
import numpy as np
import pygame
import yaml
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim.space_env import SpaceEnvironment
from prediction.lstm_predictor import PhysicsPredictor
from visualization.renderer import SpaceRenderer


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "default_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class ManualCaptureGame:
    """手动捕捉游戏模式"""

    def __init__(self):
        self.config = load_config()
        self.env = SpaceEnvironment(self.config)
        self.renderer = SpaceRenderer(
            width=self.config["display"]["width"],
            height=self.config["display"]["height"],
            star_count=self.config["display"]["star_count"]
        )

        # 物理预测器（给玩家提供辅助预测线）
        self.predictor = PhysicsPredictor(
            prediction_horizon=self.config["prediction"]["prediction_horizon"],
            dt=self.config["simulation"]["dt"]
        )

        # 游戏状态
        self.score = 0
        self.level = 1
        self.captures = 0
        self.attempts = 0
        self.show_assist = True  # 辅助预测线

        # 手动控制目标
        self.manual_target = np.array([2.0, 0.0, 1.0])
        self.move_speed = 0.08

        # 难度参数
        self.level_params = [
            {"speed": 0.2, "ang_speed": 0.02, "shape": "satellite"},
            {"speed": 0.4, "ang_speed": 0.05, "shape": "satellite"},
            {"speed": 0.6, "ang_speed": 0.1, "shape": "rocket_stage"},
            {"speed": 0.8, "ang_speed": 0.15, "shape": "fragment"},
            {"speed": 1.0, "ang_speed": 0.2, "shape": "satellite"},
            {"speed": 1.2, "ang_speed": 0.25, "shape": "rocket_stage"},
            {"speed": 1.5, "ang_speed": 0.3, "shape": "fragment"},
            {"speed": 1.8, "ang_speed": 0.4, "shape": "satellite"},
            {"speed": 2.0, "ang_speed": 0.5, "shape": "rocket_stage"},
        ]

    def get_level_params(self) -> dict:
        idx = min(self.level - 1, len(self.level_params) - 1)
        return self.level_params[idx]

    def reset_round(self):
        """重置当前回合"""
        params = self.get_level_params()
        self.env.reset(
            linear_speed=params["speed"],
            angular_speed=params["ang_speed"],
            shape=params["shape"]
        )
        self.renderer.clear_trails()
        self.manual_target = np.array([2.0, 0.0, 1.0])
        self.attempts += 1

    def run(self):
        """主游戏循环"""
        self.reset_round()
        running = True
        gripper_pressed = False

        # 开场画面
        self._show_title_screen()

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
                            if self.env.success:
                                self.captures += 1
                                self.score += self._calculate_score()
                                if self.captures % 3 == 0:
                                    self.level = min(self.level + 1, len(self.level_params))
                            self.reset_round()
                    elif event.key == pygame.K_h:
                        self.show_assist = not self.show_assist
                    elif event.key == pygame.K_r:
                        self.reset_round()

            if self.env.done:
                self._render_game(gripper_pressed)
                continue

            # 手动控制
            keys = pygame.key.get_pressed()
            gripper_pressed = keys[pygame.K_g]

            # WASD + QE 控制末端位置
            if keys[pygame.K_w]:
                self.manual_target[1] += self.move_speed
            if keys[pygame.K_s]:
                self.manual_target[1] -= self.move_speed
            if keys[pygame.K_a]:
                self.manual_target[0] -= self.move_speed
            if keys[pygame.K_d]:
                self.manual_target[0] += self.move_speed
            if keys[pygame.K_q]:
                self.manual_target[2] += self.move_speed
            if keys[pygame.K_e]:
                self.manual_target[2] -= self.move_speed

            # 限制在工作空间内
            arm_reach = sum(self.env.arm.link_lengths)
            dist = np.linalg.norm(self.manual_target - self.env.arm.base_position)
            if dist > arm_reach * 0.95:
                direction = self.manual_target - self.env.arm.base_position
                self.manual_target = self.env.arm.base_position + direction / dist * arm_reach * 0.95

            # 快捷键: F键自动追踪碎片位置
            if keys[pygame.K_f] and self.env.debris:
                self.manual_target = self.env.debris.position.copy()

            # 仿真步进
            info, done, success = self.env.step(self.manual_target, gripper_pressed)

            # 更新轨迹
            self.renderer.update_trail(
                self.env.debris.position,
                self.env.arm.get_tip_position()
            )

            # 相机控制
            key_dict = {k: keys[k] for k in [pygame.K_LEFT, pygame.K_RIGHT,
                                               pygame.K_UP, pygame.K_DOWN,
                                               pygame.K_PAGEUP, pygame.K_PAGEDOWN]}
            self.renderer.update_camera(key_dict)

            # 相机跟踪
            if self.env.debris:
                mid = (self.env.arm.get_tip_position() + self.env.debris.position) / 2
                self.renderer.camera_target = self.renderer.camera_target * 0.95 + mid * 0.05

            self._render_game(gripper_pressed)

        self.renderer.close()

    def _show_title_screen(self):
        """开场画面"""
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.renderer.close()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False

            self.renderer.draw_background()

            # 标题
            title = self.renderer.font_title.render(
                "SPACE DEBRIS CAPTURE", True, self.renderer.CYAN
            )
            rect = title.get_rect(center=(self.renderer.width // 2, 200))
            self.renderer.screen.blit(title, rect)

            subtitle = self.renderer.font_large.render(
                "Manual Control Mode", True, self.renderer.WHITE
            )
            rect2 = subtitle.get_rect(center=(self.renderer.width // 2, 250))
            self.renderer.screen.blit(subtitle, rect2)

            # 操作说明
            controls = [
                "WASD: Move arm (XY plane)",
                "Q/E: Move arm (Up/Down)",
                "G: Hold to close gripper",
                "F: Auto-track debris position",
                "H: Toggle prediction assist",
                "Arrow Keys: Rotate camera",
                "SPACE: Next round after capture",
                "R: Reset current round",
                "ESC: Quit",
            ]
            y = 320
            for text in controls:
                surf = self.renderer.font_small.render(text, True, (180, 180, 180))
                rect = surf.get_rect(center=(self.renderer.width // 2, y))
                self.renderer.screen.blit(surf, rect)
                y += 25

            start_text = self.renderer.font_large.render(
                "Press any key to start", True, self.renderer.YELLOW
            )
            rect3 = start_text.get_rect(center=(self.renderer.width // 2, y + 40))
            self.renderer.screen.blit(start_text, rect3)

            self.renderer.flip(30)

    def _render_game(self, gripper_pressed: bool):
        """渲染游戏画面"""
        self.renderer.draw_background()
        self.renderer.draw_grid()
        self.renderer.draw_axes(np.zeros(3), 1.0)

        # 轨迹
        self.renderer.draw_trail(self.renderer.debris_trail, (255, 100, 100))
        self.renderer.draw_trail(self.renderer.arm_trail, (100, 200, 255))

        # 辅助预测线
        if self.show_assist and self.env.debris:
            obs_seq = self.env.sensor.get_observation_sequence(
                self.config["prediction"]["sequence_length"]
            )
            if obs_seq is not None:
                pred_pos, pred_conf = self.predictor.predict_position_only(obs_seq)
                self.renderer.draw_predicted_trajectory(pred_pos, pred_conf)

        # 碎片
        if self.env.debris:
            self.renderer.draw_debris(
                self.env.debris.position,
                self.env.debris.orientation,
                self.env.debris.size,
                self.env.debris.shape
            )

        # 机械臂
        joint_positions = self.env.arm.get_joint_positions()
        self.renderer.draw_robot_arm(
            joint_positions,
            not gripper_pressed,
            highlight=gripper_pressed
        )

        # 目标标记
        target_2d = self.renderer.project_3d_to_2d(self.manual_target)
        pygame.draw.circle(self.renderer.screen, self.renderer.YELLOW,
                          (target_2d[0], target_2d[1]), 6, 2)

        # 连接线
        if self.env.debris:
            self.renderer.draw_connection_line(
                self.env.arm.get_tip_position(),
                self.env.debris.position
            )

        # HUD
        info = self.env._get_info()
        self.renderer.draw_hud(info, mode="MANUAL")

        # 游戏信息面板
        self._draw_game_panel()

        # 操作提示
        instructions = [
            "WASD: Move | Q/E: Up/Down | G: Gripper | F: Auto-track | H: Assist",
            "Arrow Keys: Camera | SPACE: Next | R: Reset | ESC: Quit",
        ]
        self.renderer.draw_instructions(instructions)

        # 结果
        if self.env.done:
            self.renderer.draw_result_overlay(self.env.success, self.env.time)
            if self.env.success:
                score_text = self.renderer.font_large.render(
                    f"+{self._calculate_score()} points!", True, self.renderer.GOLD
                )
                rect = score_text.get_rect(center=(self.renderer.width // 2,
                                                    self.renderer.height // 2 + 90))
                self.renderer.screen.blit(score_text, rect)

        self.renderer.flip(self.config["display"]["fps"])

    def _draw_game_panel(self):
        """绘制游戏信息面板"""
        panel_x = self.renderer.width - 250
        panel_y = 250
        panel_w = 240
        panel_h = 160

        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 160))
        self.renderer.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.renderer.screen, self.renderer.GOLD,
                        (panel_x, panel_y, panel_w, panel_h), 1)

        y = panel_y + 10
        lines = [
            (f"LEVEL: {self.level}", self.renderer.GOLD),
            (f"SCORE: {self.score}", self.renderer.WHITE),
            (f"CAPTURES: {self.captures}/{self.attempts}", self.renderer.WHITE),
            (f"Debris: {self.get_level_params()['shape']}", self.renderer.CYAN),
            (f"Speed: {self.get_level_params()['speed']:.1f} m/s", self.renderer.CYAN),
            (f"Assist: {'ON' if self.show_assist else 'OFF'} [H]", self.renderer.YELLOW),
        ]

        for text, color in lines:
            surf = self.renderer.font_small.render(text, True, color)
            self.renderer.screen.blit(surf, (panel_x + 10, y))
            y += 22

    def _calculate_score(self) -> int:
        """计算得分"""
        params = self.get_level_params()
        base_score = 100
        speed_bonus = int(params["speed"] * 200)
        time_bonus = max(0, int((30 - self.env.time) * 10))
        return base_score + speed_bonus + time_bonus


if __name__ == "__main__":
    game = ManualCaptureGame()
    game.run()
