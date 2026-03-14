"""
3D可视化渲染模块 (基于Pygame的2.5D投影渲染)
渲染太空环境、机械臂、碎片运动和抓取过程
"""
import numpy as np
import pygame
from typing import List, Optional, Tuple, Dict
import math


class SpaceRenderer:
    """太空场景渲染器"""

    # 颜色定义
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (100, 100, 100)
    DARK_GRAY = (40, 40, 40)
    RED = (255, 60, 60)
    GREEN = (60, 255, 60)
    BLUE = (60, 120, 255)
    YELLOW = (255, 255, 60)
    ORANGE = (255, 160, 40)
    CYAN = (60, 255, 255)
    PURPLE = (180, 60, 255)
    GOLD = (255, 215, 0)

    def __init__(self, width: int = 1200, height: int = 800, star_count: int = 200):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Space Debris Capture Simulation - 太空垃圾捕获仿真")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)

        # 相机参数
        self.camera_distance = 15.0
        self.camera_angle_h = 0.3   # 水平角
        self.camera_angle_v = 0.4   # 垂直角
        self.camera_target = np.zeros(3)

        # 星空背景
        self.stars = [(np.random.randint(0, width),
                       np.random.randint(0, height),
                       np.random.randint(1, 3),
                       np.random.randint(150, 255))
                      for _ in range(star_count)]

        # 轨迹缓存
        self.debris_trail: List[np.ndarray] = []
        self.arm_trail: List[np.ndarray] = []
        self.predicted_trail: List[np.ndarray] = []
        self.max_trail = 200

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int, float]:
        """3D到2D投影 (简单透视投影)"""
        # 相机变换
        ch, cv = self.camera_angle_h, self.camera_angle_v
        p = point_3d - self.camera_target

        # 旋转
        cos_h, sin_h = np.cos(ch), np.sin(ch)
        cos_v, sin_v = np.cos(cv), np.sin(cv)

        x = p[0] * cos_h - p[1] * sin_h
        y_temp = p[0] * sin_h + p[1] * cos_h
        y = y_temp * cos_v - p[2] * sin_v
        z = y_temp * sin_v + p[2] * cos_v + self.camera_distance

        if z < 0.1:
            z = 0.1

        # 透视投影
        fov = 500
        sx = int(self.width / 2 + fov * x / z)
        sy = int(self.height / 2 - fov * y / z)

        return sx, sy, z

    def draw_background(self):
        """绘制星空背景"""
        self.screen.fill(self.BLACK)
        for sx, sy, size, brightness in self.stars:
            color = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, color, (sx, sy), size)

    def draw_grid(self):
        """绘制参考网格"""
        grid_size = 10
        grid_step = 2
        for i in range(-grid_size, grid_size + 1, grid_step):
            # X方向线
            p1 = self.project_3d_to_2d(np.array([i, -grid_size, 0.0]))
            p2 = self.project_3d_to_2d(np.array([i, grid_size, 0.0]))
            pygame.draw.line(self.screen, (20, 30, 40), (p1[0], p1[1]), (p2[0], p2[1]), 1)
            # Y方向线
            p1 = self.project_3d_to_2d(np.array([-grid_size, i, 0.0]))
            p2 = self.project_3d_to_2d(np.array([grid_size, i, 0.0]))
            pygame.draw.line(self.screen, (20, 30, 40), (p1[0], p1[1]), (p2[0], p2[1]), 1)

    def draw_axes(self, origin: np.ndarray, length: float = 1.0):
        """绘制坐标轴"""
        o = self.project_3d_to_2d(origin)
        px = self.project_3d_to_2d(origin + np.array([length, 0, 0]))
        py = self.project_3d_to_2d(origin + np.array([0, length, 0]))
        pz = self.project_3d_to_2d(origin + np.array([0, 0, length]))

        pygame.draw.line(self.screen, self.RED, (o[0], o[1]), (px[0], px[1]), 2)
        pygame.draw.line(self.screen, self.GREEN, (o[0], o[1]), (py[0], py[1]), 2)
        pygame.draw.line(self.screen, self.BLUE, (o[0], o[1]), (pz[0], pz[1]), 2)

    def draw_robot_arm(self, joint_positions: List[np.ndarray],
                        gripper_open: bool = True, highlight: bool = False):
        """绘制机械臂"""
        if len(joint_positions) < 2:
            return

        projected = [self.project_3d_to_2d(p) for p in joint_positions]

        # 绘制连杆
        for i in range(len(projected) - 1):
            p1 = projected[i]
            p2 = projected[i + 1]

            # 连杆颜色渐变
            t = i / max(1, len(projected) - 2)
            color = (
                int(60 + 140 * t),
                int(180 - 80 * t),
                int(255 - 100 * t)
            )
            width = max(2, int(6 - i * 0.5))
            pygame.draw.line(self.screen, color, (p1[0], p1[1]), (p2[0], p2[1]), width)

        # 绘制关节
        for i, p in enumerate(projected):
            radius = max(3, 8 - i)
            color = self.GOLD if highlight else self.CYAN
            pygame.draw.circle(self.screen, color, (p[0], p[1]), radius)
            pygame.draw.circle(self.screen, self.WHITE, (p[0], p[1]), radius, 1)

        # 绘制末端执行器
        tip = projected[-1]
        if gripper_open:
            # 张开的夹爪
            for angle in [-0.4, 0, 0.4]:
                dx = int(12 * np.cos(angle))
                dy = int(12 * np.sin(angle))
                pygame.draw.line(self.screen, self.YELLOW,
                               (tip[0], tip[1]), (tip[0] + dx, tip[1] + dy), 3)
        else:
            # 闭合的夹爪
            pygame.draw.circle(self.screen, self.GREEN, (tip[0], tip[1]), 8)
            pygame.draw.circle(self.screen, self.WHITE, (tip[0], tip[1]), 8, 2)

    def draw_debris(self, position: np.ndarray, orientation: np.ndarray,
                     size: np.ndarray, shape: str = "satellite"):
        """绘制太空碎片"""
        center = self.project_3d_to_2d(position)

        # 根据距离计算显示大小
        scale = max(5, int(40 / max(1, center[2] / 5)))

        if shape == "satellite":
            self._draw_satellite(center, scale, orientation)
        elif shape == "rocket_stage":
            self._draw_rocket_stage(center, scale, orientation)
        else:
            self._draw_fragment(center, scale, orientation)

    def _draw_satellite(self, center: Tuple, scale: int, orientation: np.ndarray):
        """绘制卫星碎片"""
        sx, sy, _ = center
        # 主体
        rect = pygame.Rect(sx - scale, sy - scale//2, scale*2, scale)
        pygame.draw.rect(self.screen, self.GRAY, rect)
        pygame.draw.rect(self.screen, self.WHITE, rect, 1)
        # 太阳能板
        pygame.draw.rect(self.screen, self.BLUE,
                        (sx - scale*2, sy - scale//4, scale, scale//2))
        pygame.draw.rect(self.screen, self.BLUE,
                        (sx + scale, sy - scale//4, scale, scale//2))
        # 天线
        pygame.draw.line(self.screen, self.WHITE,
                        (sx, sy - scale//2), (sx, sy - scale), 2)
        pygame.draw.circle(self.screen, self.RED, (sx, sy - scale), 3)

    def _draw_rocket_stage(self, center: Tuple, scale: int, orientation: np.ndarray):
        """绘制火箭残骸"""
        sx, sy, _ = center
        # 圆柱体 (简化为矩形)
        rect = pygame.Rect(sx - scale//3, sy - scale, scale*2//3, scale*2)
        pygame.draw.rect(self.screen, (150, 150, 160), rect)
        pygame.draw.rect(self.screen, self.WHITE, rect, 1)
        # 喷口
        pygame.draw.ellipse(self.screen, self.ORANGE,
                           (sx - scale//2, sy + scale - 4, scale, 8))

    def _draw_fragment(self, center: Tuple, scale: int, orientation: np.ndarray):
        """绘制碎片"""
        sx, sy, _ = center
        # 不规则多边形
        n_points = 6
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points + orientation[1] * 3
            r = scale * (0.5 + 0.5 * abs(np.sin(angle * 2 + orientation[2])))
            points.append((sx + int(r * np.cos(angle)),
                          sy + int(r * np.sin(angle))))
        pygame.draw.polygon(self.screen, (120, 120, 130), points)
        pygame.draw.polygon(self.screen, self.WHITE, points, 1)

    def draw_trail(self, trail: List[np.ndarray], color: Tuple, max_points: int = 100):
        """绘制运动轨迹"""
        if len(trail) < 2:
            return
        points = trail[-max_points:]
        for i in range(1, len(points)):
            alpha = i / len(points)
            c = tuple(int(c * alpha) for c in color)
            p1 = self.project_3d_to_2d(points[i-1])
            p2 = self.project_3d_to_2d(points[i])
            pygame.draw.line(self.screen, c, (p1[0], p1[1]), (p2[0], p2[1]), 1)

    def draw_predicted_trajectory(self, predictions: np.ndarray,
                                    confidence: Optional[np.ndarray] = None):
        """绘制预测轨迹"""
        if len(predictions) < 2:
            return
        for i in range(1, len(predictions)):
            alpha = confidence[i] if confidence is not None else 0.5
            c = (int(255 * alpha), int(255 * (1-alpha)), 0)
            p1 = self.project_3d_to_2d(predictions[i-1])
            p2 = self.project_3d_to_2d(predictions[i])
            pygame.draw.line(self.screen, c, (p1[0], p1[1]), (p2[0], p2[1]), 2)

        # 预测终点标记
        end = self.project_3d_to_2d(predictions[-1])
        pygame.draw.circle(self.screen, self.YELLOW, (end[0], end[1]), 5, 2)

    def draw_connection_line(self, pos1: np.ndarray, pos2: np.ndarray):
        """绘制两点之间的虚线连接"""
        p1 = self.project_3d_to_2d(pos1)
        p2 = self.project_3d_to_2d(pos2)
        dist = np.linalg.norm(pos1 - pos2)

        color = self.GREEN if dist < 0.5 else self.YELLOW if dist < 2.0 else self.RED
        # 虚线
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max(1, int(np.sqrt(dx**2 + dy**2)))
        for i in range(0, length, 8):
            t1 = i / length
            t2 = min(1.0, (i + 4) / length)
            x1 = int(p1[0] + dx * t1)
            y1 = int(p1[1] + dy * t1)
            x2 = int(p1[0] + dx * t2)
            y2 = int(p1[1] + dy * t2)
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 1)

    def draw_hud(self, info: Dict, mode: str = "AUTO"):
        """绘制HUD信息面板"""
        # 左上角 - 状态信息
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 280, 220
        panel_surface = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 160))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, self.CYAN, (panel_x, panel_y, panel_w, panel_h), 1)

        y = panel_y + 8
        mode_color = self.GREEN if mode == "AUTO" else self.YELLOW
        mode_text = self.font_large.render(f"MODE: {mode}", True, mode_color)
        self.screen.blit(mode_text, (panel_x + 10, y))
        y += 28

        lines = [
            (f"Time: {info.get('time', 0):.2f}s", self.WHITE),
            (f"Distance: {info.get('distance', 0):.3f}m", self.WHITE),
            (f"Debris Speed: {np.linalg.norm(info.get('debris_velocity', [0,0,0])):.2f}m/s", self.WHITE),
            (f"Manipulability: {info.get('manipulability', 0):.3f}", self.WHITE),
            (f"Gripper: {'OPEN' if info.get('gripper_open', True) else 'CLOSED'}",
             self.YELLOW if info.get('gripper_open', True) else self.GREEN),
        ]

        for text, color in lines:
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (panel_x + 10, y))
            y += 22

        # 右上角 - 距离指示器
        self._draw_distance_indicator(info.get('distance', 10))

    def _draw_distance_indicator(self, distance: float):
        """距离指示器"""
        x = self.width - 60
        y = 30
        h = 200

        # 背景条
        pygame.draw.rect(self.screen, self.DARK_GRAY, (x - 10, y, 40, h))
        pygame.draw.rect(self.screen, self.WHITE, (x - 10, y, 40, h), 1)

        # 距离标尺
        max_dist = 15.0
        fill_h = int(h * min(1.0, distance / max_dist))
        color = self.GREEN if distance < 0.5 else self.YELLOW if distance < 3.0 else self.RED
        pygame.draw.rect(self.screen, color, (x - 8, y + h - fill_h, 36, fill_h))

        # 标签
        label = self.font_small.render(f"{distance:.1f}m", True, self.WHITE)
        self.screen.blit(label, (x - 15, y + h + 5))

    def draw_instructions(self, instructions: List[str]):
        """绘制操作说明"""
        y = self.height - len(instructions) * 22 - 10
        for text in instructions:
            surf = self.font_small.render(text, True, (150, 150, 150))
            self.screen.blit(surf, (10, y))
            y += 22

    def draw_result_overlay(self, success: bool, time_taken: float):
        """绘制结果覆盖层"""
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        if success:
            text = "CAPTURE SUCCESSFUL!"
            color = self.GREEN
        else:
            text = "CAPTURE FAILED"
            color = self.RED

        title = self.font_title.render(text, True, color)
        rect = title.get_rect(center=(self.width // 2, self.height // 2 - 30))
        self.screen.blit(title, rect)

        time_text = self.font_large.render(f"Time: {time_taken:.2f}s", True, self.WHITE)
        rect2 = time_text.get_rect(center=(self.width // 2, self.height // 2 + 20))
        self.screen.blit(time_text, rect2)

        hint = self.font_small.render("Press SPACE to restart, ESC to quit", True, self.GRAY)
        rect3 = hint.get_rect(center=(self.width // 2, self.height // 2 + 60))
        self.screen.blit(hint, rect3)

    def update_camera(self, keys_pressed: dict):
        """根据输入更新相机"""
        speed = 0.03
        if keys_pressed.get(pygame.K_LEFT):
            self.camera_angle_h -= speed
        if keys_pressed.get(pygame.K_RIGHT):
            self.camera_angle_h += speed
        if keys_pressed.get(pygame.K_UP):
            self.camera_angle_v = min(1.2, self.camera_angle_v + speed)
        if keys_pressed.get(pygame.K_DOWN):
            self.camera_angle_v = max(-0.5, self.camera_angle_v - speed)
        if keys_pressed.get(pygame.K_PAGEUP):
            self.camera_distance = max(5, self.camera_distance - 0.2)
        if keys_pressed.get(pygame.K_PAGEDOWN):
            self.camera_distance = min(30, self.camera_distance + 0.2)

    def update_trail(self, debris_pos: np.ndarray, arm_tip_pos: np.ndarray):
        """更新轨迹"""
        self.debris_trail.append(debris_pos.copy())
        self.arm_trail.append(arm_tip_pos.copy())
        if len(self.debris_trail) > self.max_trail:
            self.debris_trail.pop(0)
        if len(self.arm_trail) > self.max_trail:
            self.arm_trail.pop(0)

    def clear_trails(self):
        self.debris_trail.clear()
        self.arm_trail.clear()
        self.predicted_trail.clear()

    def flip(self, fps: int = 60):
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
