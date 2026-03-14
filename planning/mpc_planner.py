"""
模型预测控制 (MPC) 轨迹规划模块
以预测的目标轨迹为参考，规划机械臂末端的最优运动轨迹
"""
import numpy as np
from scipy.optimize import minimize
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.robot_arm import RobotArm


class MPCPlanner:
    """MPC轨迹规划器"""

    def __init__(self, arm: RobotArm, config: dict):
        self.arm = arm
        mpc_cfg = config["mpc"]

        self.N_pred = mpc_cfg["prediction_horizon"]   # 预测时域
        self.N_ctrl = mpc_cfg["control_horizon"]       # 控制时域
        self.w_pos = mpc_cfg["weight_position"]        # 位置跟踪权重
        self.w_vel = mpc_cfg["weight_velocity"]        # 速度匹配权重
        self.w_torque = mpc_cfg["weight_torque"]       # 力矩平滑权重
        self.w_singular = mpc_cfg["weight_singularity"] # 奇异位形惩罚

        self.dt = config["simulation"]["dt"]
        self.n_joints = arm.n_joints

        # 上一次的最优控制序列（用于热启动）
        self.last_solution = None

    def plan(self, predicted_trajectory: np.ndarray,
             predicted_velocity: Optional[np.ndarray] = None,
             confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        基于预测轨迹规划机械臂运动
        predicted_trajectory: (N, 3) 预测的目标位置序列
        predicted_velocity: (N, 3) 预测的目标速度序列
        confidence: (N,) 预测置信度
        返回: target_position (3,) 当前步的末端目标位置
        """
        if confidence is None:
            confidence = np.ones(len(predicted_trajectory))

        current_tip = self.arm.get_tip_position()

        # 策略：用近期高置信度预测做提前量补偿
        # 选择前10步中置信度加权的最佳目标点
        lookahead = min(10, len(predicted_trajectory))
        best_idx = 0
        best_score = -float('inf')

        arm_reach = sum(self.arm.link_lengths)

        for i in range(lookahead):
            pos = predicted_trajectory[i]
            dist = np.linalg.norm(pos - current_tip)

            if dist > arm_reach * 0.95:
                continue

            reachability = 1.0 - dist / arm_reach
            score = reachability * 2.0 + confidence[i] * 3.0 + np.exp(-0.15 * i) * 1.0

            if score > best_score:
                best_score = score
                best_idx = i

        intercept_pos = predicted_trajectory[best_idx]

        # 生成平滑轨迹
        current_vel = self.arm.get_tip_velocity()
        target = self._smooth_trajectory_to_target(
            current_tip, current_vel, intercept_pos, max(best_idx, 1)
        )

        return target

    def _find_intercept_point(self, predicted_trajectory: np.ndarray,
                               confidence: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        找到最佳拦截点 - 综合考虑可达性、置信度和时间
        """
        tip_pos = self.arm.get_tip_position()
        best_score = -float('inf')
        best_idx = 0
        best_pos = predicted_trajectory[0]

        arm_reach = sum(self.arm.link_lengths)

        for i in range(len(predicted_trajectory)):
            pos = predicted_trajectory[i]
            dist_to_target = np.linalg.norm(pos - tip_pos)

            # 可达性评分 (在臂展范围内得分高)
            if dist_to_target > arm_reach * 0.95:
                reachability = 0.0
            else:
                reachability = 1.0 - dist_to_target / arm_reach

            # 时间评分 (越早越好，但不能太早)
            time_score = np.exp(-0.1 * i) if i > 3 else 0.3

            # 综合评分
            score = (reachability * 3.0 +
                     confidence[i] * 4.0 +
                     time_score * 2.0)

            if score > best_score and reachability > 0.1:
                best_score = score
                best_idx = i
                best_pos = pos

        return best_idx, best_pos

    def _smooth_trajectory_to_target(self, current_pos: np.ndarray,
                                      current_vel: np.ndarray,
                                      target_pos: np.ndarray,
                                      steps_to_target: int) -> np.ndarray:
        """
        生成平滑的末端轨迹到目标点
        使用三次多项式插值确保速度连续
        """
        if steps_to_target <= 1:
            return target_pos

        # 归一化时间参数 t ∈ [0, 1]
        # 当前步对应 t = 1/steps_to_target
        t = min(1.0, 1.0 / max(1, steps_to_target))

        # 三次Hermite插值
        # p(t) = (2t³-3t²+1)p0 + (t³-2t²+t)v0 + (-2t³+3t²)p1 + (t³-t²)v1
        T = max(steps_to_target * self.dt, 0.01)
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = (t**3 - 2*t**2 + t) * T
        h01 = -2*t**3 + 3*t**2
        h11 = (t**3 - t**2) * T

        # 目标速度设为零（抓取时刻相对速度最小化）
        target_vel = np.zeros(3)

        next_pos = h00 * current_pos + h10 * current_vel + h01 * target_pos + h11 * target_vel

        return next_pos

    def plan_with_optimization(self, predicted_trajectory: np.ndarray,
                                confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        完整MPC优化版本 - 优化未来N步的关节角度序列
        """
        if confidence is None:
            confidence = np.ones(len(predicted_trajectory))

        n_vars = self.N_ctrl * self.n_joints
        current_angles = self.arm.joint_angles.copy()

        # 初始猜测
        if self.last_solution is not None:
            x0 = np.roll(self.last_solution.reshape(self.N_ctrl, -1), -1, axis=0).flatten()
        else:
            x0 = np.tile(current_angles, self.N_ctrl)

        # 优化
        bounds = [(-self.arm.joint_max_velocity * self.dt,
                   self.arm.joint_max_velocity * self.dt)] * n_vars

        def cost(x):
            return self._mpc_cost(x, current_angles, predicted_trajectory, confidence)

        result = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-6})

        self.last_solution = result.x

        # 返回第一步的目标关节角度增量
        dq = result.x[:self.n_joints]
        target_angles = current_angles + dq
        target_pos = self.arm.get_tip_position(target_angles)

        return target_pos

    def _mpc_cost(self, x: np.ndarray, current_angles: np.ndarray,
                   predicted_trajectory: np.ndarray,
                   confidence: np.ndarray) -> float:
        """MPC代价函数"""
        dq_sequence = x.reshape(self.N_ctrl, self.n_joints)
        cost = 0.0
        angles = current_angles.copy()

        for k in range(self.N_ctrl):
            dq = dq_sequence[k]
            angles = angles + dq

            # 正运动学
            tip_pos = self.arm.get_tip_position(angles)

            # 位置跟踪误差
            if k < len(predicted_trajectory):
                pos_error = np.linalg.norm(tip_pos - predicted_trajectory[k])
                conf = confidence[k] if k < len(confidence) else 0.5
                cost += self.w_pos * conf * pos_error**2

            # 力矩平滑 (关节角度变化量)
            cost += self.w_torque * np.sum(dq**2)

            # 奇异位形惩罚
            J = self.arm.jacobian(angles)
            svd_min = np.linalg.svd(J, compute_uv=False)[-1]
            if svd_min < 0.01:
                cost += self.w_singular * (1.0 / (svd_min + 1e-6))

        return cost

    def reset(self):
        self.last_solution = None
