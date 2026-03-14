"""
真3D OpenGL渲染器 - 太空垃圾捕获仿真
PyOpenGL + Pygame 实现：真透视相机、光照、立体模型、粒子特效、星空天球
"""
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
import math
from typing import List, Optional, Tuple, Dict


class ParticleSystem:
    """粒子系统 - 用于推进器尾焰、抓取火花等特效"""

    def __init__(self, max_particles: int = 200):
        self.max_particles = max_particles
        self.positions = np.zeros((max_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 3), dtype=np.float32)
        self.colors = np.zeros((max_particles, 4), dtype=np.float32)
        self.lifetimes = np.zeros(max_particles, dtype=np.float32)
        self.alive = np.zeros(max_particles, dtype=bool)
        self.next_idx = 0

    def emit(self, position: np.ndarray, velocity: np.ndarray,
             color: Tuple[float, ...] = (1.0, 0.8, 0.2, 1.0),
             count: int = 5, spread: float = 0.3, lifetime: float = 1.0):
        for _ in range(count):
            idx = self.next_idx % self.max_particles
            self.positions[idx] = position + np.random.randn(3) * 0.05
            self.velocities[idx] = velocity + np.random.randn(3) * spread
            self.colors[idx] = color
            self.lifetimes[idx] = lifetime
            self.alive[idx] = True
            self.next_idx += 1

    def update(self, dt: float):
        mask = self.alive
        self.positions[mask] += self.velocities[mask] * dt
        self.lifetimes[mask] -= dt
        self.colors[mask, 3] = np.clip(self.lifetimes[mask], 0, 1)
        self.alive[self.lifetimes <= 0] = False

    def draw(self):
        if not np.any(self.alive):
            return
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glPointSize(4.0)
        glBegin(GL_POINTS)
        for i in range(self.max_particles):
            if self.alive[i]:
                glColor4f(*self.colors[i])
                glVertex3f(*self.positions[i])
        glEnd()
        glPopAttrib()


class SpaceRenderer3D:
    """真3D太空场景渲染器"""

    def __init__(self, width: int = 1280, height: int = 800, star_count: int = 2000):
        pygame.init()
        self.width = width
        self.height = height
        pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Space Debris Capture 3D - 太空垃圾捕获仿真")
        self.clock = pygame.time.Clock()

        # 初始化OpenGL
        self._init_gl()

        # 相机参数
        self.cam_distance = 12.0
        self.cam_yaw = 30.0      # 水平角度
        self.cam_pitch = 25.0    # 俯仰角度
        self.cam_target = np.array([0.0, 0.0, 0.0])
        self.cam_smooth_target = np.array([0.0, 0.0, 0.0])

        # 星空
        self.star_count = star_count
        self._generate_stars()

        # 粒子系统
        self.thruster_particles = ParticleSystem(300)
        self.spark_particles = ParticleSystem(200)

        # 轨迹
        self.debris_trail: List[np.ndarray] = []
        self.arm_trail: List[np.ndarray] = []
        self.max_trail = 300

        # 显示列表缓存
        self._build_display_lists()

        # HUD用的pygame surface
        self.hud_font_large = pygame.font.SysFont("consolas", 20)
        self.hud_font_small = pygame.font.SysFont("consolas", 14)
        self.hud_font_title = pygame.font.SysFont("consolas", 28, bold=True)

    def _init_gl(self):
        """初始化OpenGL状态"""
        glClearColor(0.0, 0.0, 0.02, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)

        # 主光源 - 太阳光（方向光）
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 10.0, 7.0, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 0.95, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.15, 1.0])

        # 补光
        glLightfv(GL_LIGHT1, GL_POSITION, [-3.0, -5.0, 2.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.2, 0.25, 0.4, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.1, 0.1, 0.2, 1.0])

        # 材质
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0)

        # 透视投影
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 500.0)
        glMatrixMode(GL_MODELVIEW)

    def _generate_stars(self):
        """生成星空天球"""
        self.stars = []
        for _ in range(self.star_count):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-np.pi / 2, np.pi / 2)
            r = 200.0
            x = r * np.cos(phi) * np.cos(theta)
            y = r * np.cos(phi) * np.sin(theta)
            z = r * np.sin(phi)
            brightness = np.random.uniform(0.3, 1.0)
            size = np.random.uniform(1.0, 3.0)
            self.stars.append((x, y, z, brightness, size))

    def _build_display_lists(self):
        """预编译显示列表"""
        # 星空显示列表
        self.star_list = glGenLists(1)
        glNewList(self.star_list, GL_COMPILE)
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        for x, y, z, b, s in self.stars:
            glPointSize(s)
            glBegin(GL_POINTS)
            glColor3f(b, b, b * 0.95)
            glVertex3f(x, y, z)
            glEnd()
        glPopAttrib()
        glEndList()

        # 单位球体
        self.sphere_quad = gluNewQuadric()
        gluQuadricNormals(self.sphere_quad, GLU_SMOOTH)

    def setup_camera(self):
        """设置相机"""
        glLoadIdentity()
        # 平滑跟踪
        self.cam_smooth_target += (self.cam_target - self.cam_smooth_target) * 0.05

        yaw_rad = math.radians(self.cam_yaw)
        pitch_rad = math.radians(self.cam_pitch)

        cam_x = self.cam_smooth_target[0] + self.cam_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cam_y = self.cam_smooth_target[1] + self.cam_distance * math.sin(pitch_rad)
        cam_z = self.cam_smooth_target[2] + self.cam_distance * math.cos(pitch_rad) * math.cos(yaw_rad)

        gluLookAt(cam_x, cam_y, cam_z,
                  self.cam_smooth_target[0], self.cam_smooth_target[1], self.cam_smooth_target[2],
                  0, 1, 0)

    def draw_stars(self):
        """绘制星空天球"""
        glPushMatrix()
        glTranslatef(*self.cam_smooth_target)
        glCallList(self.star_list)
        glPopMatrix()

    def draw_grid(self):
        """绘制参考网格"""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        grid_size = 20
        step = 2
        for i in range(-grid_size, grid_size + 1, step):
            alpha = 0.08 if i % 4 != 0 else 0.15
            glColor4f(0.3, 0.4, 0.6, alpha)
            glVertex3f(i, -0.01, -grid_size)
            glVertex3f(i, -0.01, grid_size)
            glVertex3f(-grid_size, -0.01, i)
            glVertex3f(grid_size, -0.01, i)
        glEnd()
        glPopAttrib()

    def draw_axes(self, origin=(0, 0, 0), length=1.5):
        """绘制坐标轴"""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glLineWidth(2.5)
        ox, oy, oz = origin
        glBegin(GL_LINES)
        # X - 红
        glColor3f(1, 0.2, 0.2)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox + length, oy, oz)
        # Y - 绿
        glColor3f(0.2, 1, 0.2)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox, oy + length, oz)
        # Z - 蓝
        glColor3f(0.3, 0.5, 1)
        glVertex3f(ox, oy, oz)
        glVertex3f(ox, oy, oz + length)
        glEnd()
        glPopAttrib()

    def draw_robot_arm(self, joint_positions: List[np.ndarray],
                       gripper_open: bool = True, highlight: bool = False):
        """绘制3D机械臂 - 圆柱连杆 + 球形关节 + 夹爪"""
        if len(joint_positions) < 2:
            return

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glEnable(GL_LIGHTING)

        for i in range(len(joint_positions) - 1):
            p1 = joint_positions[i]
            p2 = joint_positions[i + 1]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue

            # 连杆颜色渐变 (银灰 → 蓝白)
            t = i / max(1, len(joint_positions) - 2)
            r = 0.5 + 0.3 * t
            g = 0.55 + 0.2 * t
            b = 0.6 + 0.4 * t

            # 绘制圆柱连杆
            self._draw_cylinder(p1, p2, radius=0.06 - i * 0.005,
                                color=(r, g, b, 1.0))

            # 绘制关节球
            joint_color = (1.0, 0.85, 0.2, 1.0) if highlight else (0.3, 0.8, 0.9, 1.0)
            glPushMatrix()
            glTranslatef(*p1)
            glColor4f(*joint_color)
            gluSphere(self.sphere_quad, 0.09 - i * 0.005, 16, 16)
            glPopMatrix()

        # 末端关节
        tip = joint_positions[-1]
        glPushMatrix()
        glTranslatef(*tip)
        if highlight:
            glColor4f(1.0, 1.0, 0.3, 1.0)
        else:
            glColor4f(0.3, 0.9, 0.9, 1.0)
        gluSphere(self.sphere_quad, 0.07, 16, 16)
        glPopMatrix()

        # 绘制夹爪
        self._draw_gripper(tip, gripper_open, highlight)

        glPopAttrib()

    def _draw_cylinder(self, p1: np.ndarray, p2: np.ndarray,
                       radius: float = 0.05, color: Tuple = (0.6, 0.6, 0.7, 1.0)):
        """绘制从p1到p2的圆柱体"""
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return

        d = direction / length

        # 计算旋转：从Z轴旋转到direction方向
        if abs(d[2] - 1.0) < 1e-6:
            angle, axis = 0, (0, 1, 0)
        elif abs(d[2] + 1.0) < 1e-6:
            angle, axis = 180, (0, 1, 0)
        else:
            angle = math.degrees(math.acos(np.clip(d[2], -1, 1)))
            axis = np.cross([0, 0, 1], d)
            axis = axis / (np.linalg.norm(axis) + 1e-8)

        glPushMatrix()
        glTranslatef(*p1)
        glRotatef(angle, *axis)
        glColor4f(*color)

        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        gluCylinder(quad, radius, radius * 0.85, length, 12, 1)
        gluDeleteQuadric(quad)

        glPopMatrix()

    def _draw_gripper(self, tip_pos: np.ndarray, gripper_open: bool, highlight: bool):
        """绘制末端夹爪"""
        glPushMatrix()
        glTranslatef(*tip_pos)

        if gripper_open:
            # 三指张开
            glColor4f(0.9, 0.9, 0.3, 1.0)
            for angle_deg in [0, 120, 240]:
                glPushMatrix()
                glRotatef(angle_deg, 0, 0, 1)
                glRotatef(30, 1, 0, 0)
                quad = gluNewQuadric()
                gluCylinder(quad, 0.02, 0.01, 0.15, 8, 1)
                gluDeleteQuadric(quad)
                glPopMatrix()
        else:
            # 闭合 - 发光球
            if highlight:
                glColor4f(0.2, 1.0, 0.3, 0.9)
            else:
                glColor4f(0.3, 0.9, 0.3, 0.9)
            gluSphere(self.sphere_quad, 0.1, 16, 16)

        glPopMatrix()

    def draw_debris(self, position: np.ndarray, orientation: np.ndarray,
                    size: np.ndarray, shape: str = "satellite"):
        """绘制3D太空碎片"""
        glPushMatrix()
        glTranslatef(*position)

        # 四元数转旋转矩阵并应用
        w, x, y, z = orientation
        rot_matrix = np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y), 0],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x), 0],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y), 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        glMultMatrixd(rot_matrix.T.flatten())

        if shape == "satellite":
            self._draw_satellite_model(size)
        elif shape == "rocket_stage":
            self._draw_rocket_model(size)
        else:
            self._draw_fragment_model(size)

        glPopMatrix()

    def _draw_satellite_model(self, size: np.ndarray):
        """绘制卫星模型 - 主体 + 太阳能板 + 天线"""
        sx, sy, sz = size * 0.3

        # 主体 (金色立方体)
        glColor4f(0.85, 0.75, 0.4, 1.0)
        self._draw_box(sx, sy, sz)

        # 太阳能板 (蓝色薄板)
        glColor4f(0.15, 0.2, 0.6, 0.95)
        glPushMatrix()
        glTranslatef(sx + sx * 0.8, 0, 0)
        self._draw_box(sx * 0.7, sy * 0.9, sz * 0.03)
        glPopMatrix()

        glPushMatrix()
        glTranslatef(-sx - sx * 0.8, 0, 0)
        self._draw_box(sx * 0.7, sy * 0.9, sz * 0.03)
        glPopMatrix()

        # 太阳能板网格线
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glColor4f(0.3, 0.4, 0.8, 0.6)
        glLineWidth(1.0)
        for side in [1, -1]:
            cx = side * (sx + sx * 0.8)
            pw, ph = sx * 0.7, sy * 0.9
            glBegin(GL_LINES)
            for i in range(5):
                t = -pw + 2 * pw * i / 4
                glVertex3f(cx + t, -ph, sz * 0.04 * side)
                glVertex3f(cx + t, ph, sz * 0.04 * side)
            for i in range(7):
                t = -ph + 2 * ph * i / 6
                glVertex3f(cx - pw, t, sz * 0.04 * side)
                glVertex3f(cx + pw, t, sz * 0.04 * side)
            glEnd()
        glPopAttrib()

        # 天线
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glColor3f(0.8, 0.8, 0.8)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(0, sy, 0)
        glVertex3f(0, sy + sz * 1.5, 0)
        glEnd()
        glPopAttrib()

        # 天线顶端小球
        glPushMatrix()
        glTranslatef(0, sy + sz * 1.5, 0)
        glColor4f(1.0, 0.3, 0.2, 1.0)
        gluSphere(self.sphere_quad, sz * 0.15, 10, 10)
        glPopMatrix()

    def _draw_rocket_model(self, size: np.ndarray):
        """绘制火箭残骸 - 圆柱体 + 锥形头 + 喷口"""
        sx, sy, sz = size * 0.25
        radius = (sx + sz) / 2
        length = sy * 2

        # 主体圆柱
        glColor4f(0.75, 0.75, 0.78, 1.0)
        glPushMatrix()
        glRotatef(-90, 1, 0, 0)
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)
        glTranslatef(0, 0, -length / 2)
        gluCylinder(quad, radius, radius, length, 20, 4)
        gluDeleteQuadric(quad)
        glPopMatrix()

        # 锥形头部
        glColor4f(0.9, 0.4, 0.2, 1.0)
        glPushMatrix()
        glTranslatef(0, length / 2, 0)
        glRotatef(-90, 1, 0, 0)
        quad = gluNewQuadric()
        gluCylinder(quad, radius, 0, radius * 1.5, 20, 4)
        gluDeleteQuadric(quad)
        glPopMatrix()

        # 喷口
        glColor4f(0.3, 0.3, 0.35, 1.0)
        glPushMatrix()
        glTranslatef(0, -length / 2, 0)
        glRotatef(90, 1, 0, 0)
        quad = gluNewQuadric()
        gluCylinder(quad, radius * 0.8, radius * 1.1, radius * 0.5, 20, 2)
        gluDeleteQuadric(quad)
        glPopMatrix()

        # 条纹装饰
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glColor4f(0.9, 0.2, 0.1, 0.8)
        glLineWidth(3.0)
        for y_off in [-length * 0.15, length * 0.15]:
            glBegin(GL_LINE_LOOP)
            for a in range(24):
                angle = 2 * math.pi * a / 24
                glVertex3f(radius * 1.01 * math.cos(angle),
                          y_off,
                          radius * 1.01 * math.sin(angle))
            glEnd()
        glPopAttrib()

    def _draw_fragment_model(self, size: np.ndarray):
        """绘制不规则碎片 - 随机变形的多面体"""
        glColor4f(0.5, 0.5, 0.55, 1.0)
        scale = np.mean(size) * 0.3

        # 用变形的二十面体近似不规则碎片
        glPushMatrix()
        glScalef(scale, scale * 0.7, scale * 0.85)

        # 简化：用多个随机朝向的三角面片
        np.random.seed(42)  # 固定种子保证每帧一致
        vertices = []
        for _ in range(8):
            v = np.random.randn(3)
            v = v / np.linalg.norm(v) * (0.8 + np.random.rand() * 0.4)
            vertices.append(v)

        glBegin(GL_TRIANGLES)
        for i in range(0, len(vertices) - 2):
            for j in range(i + 1, len(vertices) - 1):
                for k in range(j + 1, len(vertices)):
                    v1, v2, v3 = vertices[i], vertices[j], vertices[k]
                    normal = np.cross(v2 - v1, v3 - v1)
                    n_len = np.linalg.norm(normal)
                    if n_len > 0.1:
                        normal /= n_len
                        glNormal3f(*normal)
                        c = 0.4 + 0.2 * abs(normal[1])
                        glColor4f(c, c, c + 0.05, 1.0)
                        glVertex3f(*v1)
                        glVertex3f(*v2)
                        glVertex3f(*v3)
        glEnd()
        glPopMatrix()

    def _draw_box(self, hx: float, hy: float, hz: float):
        """绘制中心在原点的长方体"""
        glBegin(GL_QUADS)
        # 前
        glNormal3f(0, 0, 1)
        glVertex3f(-hx, -hy, hz); glVertex3f(hx, -hy, hz)
        glVertex3f(hx, hy, hz); glVertex3f(-hx, hy, hz)
        # 后
        glNormal3f(0, 0, -1)
        glVertex3f(hx, -hy, -hz); glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, hy, -hz); glVertex3f(hx, hy, -hz)
        # 上
        glNormal3f(0, 1, 0)
        glVertex3f(-hx, hy, hz); glVertex3f(hx, hy, hz)
        glVertex3f(hx, hy, -hz); glVertex3f(-hx, hy, -hz)
        # 下
        glNormal3f(0, -1, 0)
        glVertex3f(-hx, -hy, -hz); glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, -hy, hz); glVertex3f(-hx, -hy, hz)
        # 右
        glNormal3f(1, 0, 0)
        glVertex3f(hx, -hy, hz); glVertex3f(hx, -hy, -hz)
        glVertex3f(hx, hy, -hz); glVertex3f(hx, hy, hz)
        # 左
        glNormal3f(-1, 0, 0)
        glVertex3f(-hx, -hy, -hz); glVertex3f(-hx, -hy, hz)
        glVertex3f(-hx, hy, hz); glVertex3f(-hx, hy, -hz)
        glEnd()

    def draw_trail(self, trail: List[np.ndarray], color: Tuple[float, ...],
                   max_points: int = 200):
        """绘制3D运动轨迹"""
        if len(trail) < 2:
            return
        points = trail[-max_points:]
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glLineWidth(2.0)
        glBegin(GL_LINE_STRIP)
        for i, p in enumerate(points):
            alpha = (i + 1) / len(points)
            glColor4f(color[0], color[1], color[2], alpha * 0.7)
            glVertex3f(*p)
        glEnd()
        glPopAttrib()

    def draw_predicted_trajectory(self, predictions: np.ndarray,
                                   confidence: Optional[np.ndarray] = None):
        """绘制预测轨迹 - 颜色反映置信度"""
        if len(predictions) < 2:
            return
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glLineWidth(2.5)

        # 主线
        glBegin(GL_LINE_STRIP)
        for i in range(len(predictions)):
            c = confidence[i] if confidence is not None else 0.5
            glColor4f(1.0 - c, c, 0.2, c * 0.8 + 0.1)
            glVertex3f(*predictions[i])
        glEnd()

        # 预测点标记
        glPointSize(5.0)
        glBegin(GL_POINTS)
        for i in range(0, len(predictions), 10):
            c = confidence[i] if confidence is not None else 0.5
            glColor4f(1.0, 1.0, 0.3, c)
            glVertex3f(*predictions[i])
        glEnd()

        # 终点标记 - 十字
        end = predictions[-1]
        s = 0.15
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 0.0, 0.8)
        glBegin(GL_LINES)
        glVertex3f(end[0] - s, end[1], end[2])
        glVertex3f(end[0] + s, end[1], end[2])
        glVertex3f(end[0], end[1] - s, end[2])
        glVertex3f(end[0], end[1] + s, end[2])
        glVertex3f(end[0], end[1], end[2] - s)
        glVertex3f(end[0], end[1], end[2] + s)
        glEnd()

        glPopAttrib()

    def draw_connection_line(self, pos1: np.ndarray, pos2: np.ndarray):
        """绘制两点之间的虚线"""
        dist = np.linalg.norm(pos1 - pos2)
        if dist < 0.5:
            color = (0.2, 1.0, 0.3, 0.6)
        elif dist < 3.0:
            color = (1.0, 1.0, 0.2, 0.5)
        else:
            color = (1.0, 0.3, 0.2, 0.4)

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor4f(*color)
        glVertex3f(*pos1)
        glVertex3f(*pos2)
        glEnd()
        glPopAttrib()

    def draw_target_marker(self, position: np.ndarray):
        """绘制目标位置标记 - 3D十字准星"""
        s = 0.2
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glColor4f(1.0, 1.0, 0.0, 0.7)
        glBegin(GL_LINES)
        glVertex3f(position[0] - s, position[1], position[2])
        glVertex3f(position[0] + s, position[1], position[2])
        glVertex3f(position[0], position[1] - s, position[2])
        glVertex3f(position[0], position[1] + s, position[2])
        glVertex3f(position[0], position[1], position[2] - s)
        glVertex3f(position[0], position[1], position[2] + s)
        glEnd()

        # 外圈
        glBegin(GL_LINE_LOOP)
        for a in range(24):
            angle = 2 * math.pi * a / 24
            glVertex3f(position[0] + s * math.cos(angle),
                      position[1] + s * math.sin(angle),
                      position[2])
        glEnd()
        glPopAttrib()

    # ==================== HUD (2D覆盖层) ====================

    def begin_2d_overlay(self):
        """切换到2D正交投影用于HUD"""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)

    def end_2d_overlay(self):
        """恢复3D投影"""
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def _render_text_to_texture(self, text: str, font, color=(255, 255, 255),
                                 bg_alpha: int = 0) -> Tuple[int, int, int]:
        """将pygame文字渲染为OpenGL纹理，返回(texture_id, w, h)"""
        surface = font.render(text, True, color)
        w, h = surface.get_size()
        # 转为RGBA
        data = pygame.image.tostring(surface, "RGBA", True)

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        return tex_id, w, h

    def _draw_text_2d(self, x: int, y: int, text: str, font=None,
                      color=(255, 255, 255)):
        """在2D覆盖层上绘制文字"""
        if font is None:
            font = self.hud_font_small
        tex_id, w, h = self._render_text_to_texture(text, font, color)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, y)
        glTexCoord2f(1, 1); glVertex2f(x + w, y)
        glTexCoord2f(1, 0); glVertex2f(x + w, y + h)
        glTexCoord2f(0, 0); glVertex2f(x, y + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex_id])

    def _draw_panel_bg(self, x: int, y: int, w: int, h: int,
                       border_color=(0.2, 0.8, 0.9, 0.8)):
        """绘制半透明面板背景"""
        glColor4f(0.0, 0.02, 0.05, 0.75)
        glBegin(GL_QUADS)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()
        # 边框
        glColor4f(*border_color)
        glLineWidth(1.5)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()

    def draw_hud(self, info: Dict, mode: str = "AUTO"):
        """绘制HUD信息面板"""
        self.begin_2d_overlay()

        # 左上角状态面板
        self._draw_panel_bg(10, 10, 280, 200)

        mode_color = (100, 255, 100) if mode == "AUTO" else (255, 255, 100)
        self._draw_text_2d(20, 18, f"MODE: {mode}", self.hud_font_large, mode_color)

        y = 48
        debris_vel = info.get('debris_velocity', np.zeros(3))
        speed = np.linalg.norm(debris_vel)
        lines = [
            (f"Time: {info.get('time', 0):.2f}s", (220, 220, 220)),
            (f"Distance: {info.get('distance', 0):.3f}m", (220, 220, 220)),
            (f"Debris Speed: {speed:.2f}m/s", (220, 220, 220)),
            (f"Manipulability: {info.get('manipulability', 0):.3f}", (220, 220, 220)),
        ]
        gripper_open = info.get('gripper_open', True)
        grip_text = "OPEN" if gripper_open else "CLOSED"
        grip_color = (255, 255, 100) if gripper_open else (100, 255, 100)
        lines.append((f"Gripper: {grip_text}", grip_color))

        for text, color in lines:
            self._draw_text_2d(20, y, text, self.hud_font_small, color)
            y += 22

        # 右上角距离指示器
        self._draw_distance_bar(info.get('distance', 10))

        self.end_2d_overlay()

    def _draw_distance_bar(self, distance: float):
        """距离指示条"""
        x = self.width - 50
        y = 30
        h = 200
        w = 25

        self._draw_panel_bg(x - 5, y - 5, w + 10, h + 30)

        max_dist = 15.0
        fill_ratio = min(1.0, distance / max_dist)
        fill_h = int(h * fill_ratio)

        if distance < 0.5:
            color = (0.2, 1.0, 0.3, 0.8)
        elif distance < 3.0:
            color = (1.0, 1.0, 0.2, 0.8)
        else:
            color = (1.0, 0.3, 0.2, 0.8)

        glColor4f(*color)
        glBegin(GL_QUADS)
        glVertex2f(x, y + h - fill_h)
        glVertex2f(x + w, y + h - fill_h)
        glVertex2f(x + w, y + h)
        glVertex2f(x, y + h)
        glEnd()

        # 边框
        glColor4f(0.5, 0.5, 0.5, 0.5)
        glLineWidth(1.0)
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y); glVertex2f(x + w, y)
        glVertex2f(x + w, y + h); glVertex2f(x, y + h)
        glEnd()

        self._draw_text_2d(x - 10, y + h + 5, f"{distance:.1f}m",
                           self.hud_font_small, (220, 220, 220))

    def draw_instructions(self, instructions: List[str]):
        """绘制底部操作说明"""
        self.begin_2d_overlay()
        y = self.height - len(instructions) * 22 - 10
        for text in instructions:
            self._draw_text_2d(10, y, text, self.hud_font_small, (140, 140, 140))
            y += 22
        self.end_2d_overlay()

    def draw_state_label(self, state: str):
        """绘制状态标签"""
        colors = {
            "SEARCHING": (150, 150, 150),
            "TRACKING": (255, 255, 60),
            "APPROACHING": (255, 160, 40),
            "GRASPING": (60, 255, 60),
            "DONE": (60, 255, 60),
        }
        self.begin_2d_overlay()
        self._draw_text_2d(10, 220, f"State: {state}",
                           self.hud_font_large, colors.get(state, (255, 255, 255)))
        self.end_2d_overlay()

    def draw_result_overlay(self, success: bool, time_taken: float):
        """绘制结果覆盖层"""
        self.begin_2d_overlay()

        # 半透明背景
        glColor4f(0, 0, 0, 0.6)
        glBegin(GL_QUADS)
        glVertex2f(0, 0); glVertex2f(self.width, 0)
        glVertex2f(self.width, self.height); glVertex2f(0, self.height)
        glEnd()

        if success:
            text = "CAPTURE SUCCESSFUL!"
            color = (60, 255, 60)
        else:
            text = "CAPTURE FAILED"
            color = (255, 60, 60)

        cx = self.width // 2
        cy = self.height // 2
        self._draw_text_2d(cx - 180, cy - 40, text, self.hud_font_title, color)
        self._draw_text_2d(cx - 80, cy + 10, f"Time: {time_taken:.2f}s",
                           self.hud_font_large, (255, 255, 255))
        self._draw_text_2d(cx - 160, cy + 50,
                           "Press SPACE to restart, ESC to quit",
                           self.hud_font_small, (150, 150, 150))

        self.end_2d_overlay()

    # ==================== 相机控制与工具方法 ====================

    def update_camera(self, keys_pressed):
        """根据输入更新相机"""
        speed = 1.5
        zoom_speed = 0.4
        if keys_pressed.get(pygame.K_LEFT):
            self.cam_yaw -= speed
        if keys_pressed.get(pygame.K_RIGHT):
            self.cam_yaw += speed
        if keys_pressed.get(pygame.K_UP):
            self.cam_pitch = min(85, self.cam_pitch + speed)
        if keys_pressed.get(pygame.K_DOWN):
            self.cam_pitch = max(-30, self.cam_pitch - speed)
        if keys_pressed.get(pygame.K_PAGEUP):
            self.cam_distance = max(3, self.cam_distance - zoom_speed)
        if keys_pressed.get(pygame.K_PAGEDOWN):
            self.cam_distance = min(40, self.cam_distance + zoom_speed)

    def update_camera_mouse(self, dx: int, dy: int, scroll: int = 0):
        """鼠标控制相机"""
        self.cam_yaw += dx * 0.3
        self.cam_pitch = np.clip(self.cam_pitch + dy * 0.3, -30, 85)
        if scroll != 0:
            self.cam_distance = np.clip(self.cam_distance - scroll * 0.5, 3, 40)

    def update_trail(self, debris_pos: np.ndarray, arm_tip_pos: np.ndarray):
        self.debris_trail.append(debris_pos.copy())
        self.arm_trail.append(arm_tip_pos.copy())
        if len(self.debris_trail) > self.max_trail:
            self.debris_trail.pop(0)
        if len(self.arm_trail) > self.max_trail:
            self.arm_trail.pop(0)

    def clear_trails(self):
        self.debris_trail.clear()
        self.arm_trail.clear()

    def begin_frame(self):
        """开始一帧"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.setup_camera()

    def end_frame(self, fps: int = 60):
        """结束一帧"""
        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()

    # ==================== 高速碎片 & 波次模式专用渲染 ====================

    def draw_speed_trail(self, position: np.ndarray, velocity: np.ndarray,
                         color: Tuple[float, ...] = (1.0, 0.5, 0.2)):
        """绘制高速运动拖尾 - 速度越快拖尾越长越亮"""
        speed = np.linalg.norm(velocity)
        if speed < 0.1:
            return

        trail_length = min(speed * 1.2, 8.0)  # 拖尾长度与速度成正比
        vel_dir = velocity / speed
        tail_pos = position - vel_dir * trail_length

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # 加法混合，更亮

        # 多层拖尾，越远越淡
        num_segments = int(trail_length * 8)
        num_segments = max(4, min(num_segments, 40))

        for layer in range(3):
            width = 3.0 - layer * 0.8
            glLineWidth(max(1.0, width))
            glBegin(GL_LINE_STRIP)
            for i in range(num_segments + 1):
                t = i / num_segments
                p = position * (1 - t) + tail_pos * t
                alpha = (1.0 - t) * (0.8 - layer * 0.25)
                brightness = 1.0 - layer * 0.3
                glColor4f(color[0] * brightness, color[1] * brightness,
                         color[2] * brightness, alpha)
                # 加一点抖动模拟热扰动
                jitter = np.random.randn(3) * 0.02 * t * speed
                glVertex3f(*(p + jitter))
            glEnd()

        # 速度线（更细更长的辅助线）
        if speed > 2.0:
            glLineWidth(1.0)
            glBegin(GL_LINES)
            glColor4f(1.0, 0.8, 0.3, 0.3)
            far_tail = position - vel_dir * trail_length * 2
            glVertex3f(*position)
            glVertex3f(*far_tail)
            glEnd()

        glPopAttrib()

    def draw_recycle_zone(self, position: np.ndarray, radius: float,
                          collected: int, pulse_time: float = 0.0):
        """绘制回收区 - 发光圆环 + 脉冲效果"""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        pulse = 0.7 + 0.3 * math.sin(pulse_time * 3.0)

        # 底部圆盘（半透明）
        glColor4f(0.1, 0.8, 0.3, 0.15 * pulse)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(*position)
        for a in range(33):
            angle = 2 * math.pi * a / 32
            glVertex3f(position[0] + radius * math.cos(angle),
                      position[1],
                      position[2] + radius * math.sin(angle))
        glEnd()

        # 外圈光环
        glLineWidth(2.5)
        glColor4f(0.2, 1.0, 0.4, 0.6 * pulse)
        glBegin(GL_LINE_LOOP)
        for a in range(48):
            angle = 2 * math.pi * a / 48
            glVertex3f(position[0] + radius * math.cos(angle),
                      position[1] + 0.05,
                      position[2] + radius * math.sin(angle))
        glEnd()

        # 内圈
        inner_r = radius * 0.6
        glColor4f(0.3, 1.0, 0.5, 0.4 * pulse)
        glBegin(GL_LINE_LOOP)
        for a in range(32):
            angle = 2 * math.pi * a / 32
            glVertex3f(position[0] + inner_r * math.cos(angle),
                      position[1] + 0.05,
                      position[2] + inner_r * math.sin(angle))
        glEnd()

        # 竖直光柱
        glLineWidth(1.5)
        glBegin(GL_LINES)
        glColor4f(0.2, 1.0, 0.4, 0.5 * pulse)
        glVertex3f(*position)
        glColor4f(0.2, 1.0, 0.4, 0.0)
        glVertex3f(position[0], position[1] + 4.0, position[2])
        glEnd()

        # 旋转标记线
        rot_angle = pulse_time * 60  # 旋转
        for i in range(4):
            a = math.radians(rot_angle + i * 90)
            glBegin(GL_LINES)
            glColor4f(0.3, 1.0, 0.5, 0.4)
            glVertex3f(position[0] + inner_r * math.cos(a),
                      position[1] + 0.1,
                      position[2] + inner_r * math.sin(a))
            glColor4f(0.3, 1.0, 0.5, 0.1)
            glVertex3f(position[0] + radius * math.cos(a),
                      position[1] + 0.1,
                      position[2] + radius * math.sin(a))
            glEnd()

        glPopAttrib()

    def draw_warning_indicator(self, direction: np.ndarray, speed: float,
                                screen_edge: bool = True):
        """绘制碎片来袭预警 - 方向箭头 + 闪烁"""
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)

        d = direction / (np.linalg.norm(direction) + 1e-8)
        arrow_start = d * 8.0
        arrow_end = d * 6.0

        # 闪烁
        blink = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() * 0.01)

        glLineWidth(3.0)
        glColor4f(1.0, 0.3, 0.1, blink)
        glBegin(GL_LINES)
        glVertex3f(*arrow_start)
        glVertex3f(*arrow_end)
        glEnd()

        # 箭头头部
        glPointSize(8.0)
        glBegin(GL_POINTS)
        glColor4f(1.0, 0.2, 0.0, blink)
        glVertex3f(*arrow_end)
        glEnd()

        glPopAttrib()

    def draw_wave_hud(self, info: Dict, mode: str = "AUTO"):
        """绘制波次模式专用HUD"""
        self.begin_2d_overlay()

        # 左上角主面板
        self._draw_panel_bg(10, 10, 300, 260)

        mode_color = (100, 255, 100) if mode == "AUTO" else (255, 255, 100)
        self._draw_text_2d(20, 18, f"MODE: {mode}", self.hud_font_large, mode_color)

        y = 48
        speed = info.get('current_debris_speed', 0)
        lines = [
            (f"Time: {info.get('time', 0):.1f}s", (220, 220, 220)),
            (f"Distance: {info.get('distance', 0):.2f}m", (220, 220, 220)),
            (f"Debris Speed: {speed:.1f} m/s", self._speed_color(speed)),
            (f"Manipulability: {info.get('manipulability', 0):.3f}", (220, 220, 220)),
        ]

        holding = info.get('holding', False)
        grip_text = "HOLDING DEBRIS" if holding else ("OPEN" if info.get('gripper_open', True) else "CLOSED")
        grip_color = (100, 255, 200) if holding else ((255, 255, 100) if info.get('gripper_open', True) else (100, 255, 100))
        lines.append((f"Gripper: {grip_text}", grip_color))

        # 波次进度
        total = info.get('wave_total', 0)
        recycled = info.get('wave_recycled', 0)
        missed = info.get('wave_missed', 0)
        remaining = total - recycled - missed
        lines.append(("", (0, 0, 0)))  # 空行
        lines.append((f"WAVE: {recycled}/{total} recycled | {missed} missed", (100, 220, 255)))
        lines.append((f"Remaining: {remaining}", (255, 200, 100)))

        shape = info.get('current_debris_shape', '')
        if shape:
            lines.append((f"Target: {shape}", (255, 180, 100)))

        for text, color in lines:
            if text:
                self._draw_text_2d(20, y, text, self.hud_font_small, color)
            y += 22

        # 右上角距离条
        self._draw_distance_bar(info.get('distance', 10))

        # 波次进度条（顶部中央）
        self._draw_wave_progress(total, recycled, missed)

        self.end_2d_overlay()

    def _speed_color(self, speed: float) -> Tuple[int, ...]:
        if speed < 1.0:
            return (100, 255, 100)
        elif speed < 3.0:
            return (255, 255, 100)
        else:
            return (255, 100, 100)

    def _draw_wave_progress(self, total: int, recycled: int, missed: int):
        """顶部波次进度条"""
        if total == 0:
            return
        bar_w = 400
        bar_h = 20
        x = (self.width - bar_w) // 2
        y = 10

        self._draw_panel_bg(x - 5, y - 5, bar_w + 10, bar_h + 10,
                            border_color=(0.5, 0.8, 1.0, 0.6))

        seg_w = bar_w / total
        for i in range(total):
            sx = x + i * seg_w
            if i < recycled:
                glColor4f(0.2, 0.9, 0.3, 0.8)  # 已回收 - 绿
            elif i < recycled + missed:
                glColor4f(0.9, 0.2, 0.2, 0.8)  # 已错过 - 红
            else:
                glColor4f(0.3, 0.3, 0.4, 0.5)  # 待处理 - 灰
            glBegin(GL_QUADS)
            glVertex2f(sx + 1, y)
            glVertex2f(sx + seg_w - 1, y)
            glVertex2f(sx + seg_w - 1, y + bar_h)
            glVertex2f(sx + 1, y + bar_h)
            glEnd()
