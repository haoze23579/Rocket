"""
LSTM运动预测模块
利用历史观测序列预测目标未来0.5-2秒的位置和姿态轨迹
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class TrajectoryLSTM(nn.Module):
    """LSTM轨迹预测网络"""

    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, prediction_horizon: int = 50):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 输出: 预测位置(3) + 速度(3) + 四元数(4) = 10
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 10 * prediction_horizon)
        )

        # 置信度输出
        self.fc_confidence = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, prediction_horizon),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, 10)
        返回: predictions (batch, horizon, 10), confidence (batch, horizon)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)

        pred = self.fc_out(last_hidden)  # (batch, horizon*10)
        pred = pred.view(-1, self.prediction_horizon, 10)

        confidence = self.fc_confidence(last_hidden)

        return pred, confidence


class MotionPredictor:
    """运动预测器 - 封装LSTM模型的推理接口"""

    def __init__(self, model_path: Optional[str] = None,
                 hidden_size: int = 128, num_layers: int = 2,
                 prediction_horizon: int = 50):
        self.prediction_horizon = prediction_horizon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TrajectoryLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon
        ).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()

        # 归一化参数
        self.input_mean = np.zeros(10)
        self.input_std = np.ones(10)

    def set_normalization(self, mean: np.ndarray, std: np.ndarray):
        self.input_mean = mean
        self.input_std = np.clip(std, 1e-6, None)

    def predict(self, observation_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入观测序列，预测未来轨迹
        observation_sequence: (seq_len, 10)
        返回: predicted_trajectory (horizon, 10), confidence (horizon,)
        """
        # 归一化
        normalized = (observation_sequence - self.input_mean) / self.input_std

        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, conf = self.model(x)

        pred_np = pred.cpu().numpy()[0]  # (horizon, 10)
        conf_np = conf.cpu().numpy()[0]  # (horizon,)

        # 反归一化
        pred_np = pred_np * self.input_std + self.input_mean

        return pred_np, conf_np

    def predict_position_only(self, observation_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """只返回位置预测"""
        pred, conf = self.predict(observation_sequence)
        return pred[:, :3], conf


class PhysicsPredictor:
    """
    基于物理模型的运动预测（作为基线对比）
    假设匀速直线运动 + 匀角速度旋转
    """

    def __init__(self, prediction_horizon: int = 50, dt: float = 0.01):
        self.prediction_horizon = prediction_horizon
        self.dt = dt

    def predict(self, observation_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基于最近观测的线性外推"""
        if len(observation_sequence) < 2:
            return np.zeros((self.prediction_horizon, 10)), np.zeros(self.prediction_horizon)

        # 用最后几帧估计速度
        n_fit = min(10, len(observation_sequence))
        recent = observation_sequence[-n_fit:]

        # 线性拟合位置
        pos = recent[:, :3]
        vel = recent[:, 3:6]
        avg_vel = np.mean(vel, axis=0)
        last_pos = observation_sequence[-1, :3]
        last_orient = observation_sequence[-1, 6:10]

        predictions = np.zeros((self.prediction_horizon, 10))
        confidence = np.zeros(self.prediction_horizon)

        for i in range(self.prediction_horizon):
            t = (i + 1) * self.dt
            predictions[i, :3] = last_pos + avg_vel * t
            predictions[i, 3:6] = avg_vel
            predictions[i, 6:10] = last_orient  # 简化：姿态不变
            # 置信度随时间衰减
            confidence[i] = np.exp(-t * 2.0)

        return predictions, confidence

    def predict_position_only(self, observation_sequence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pred, conf = self.predict(observation_sequence)
        return pred[:, :3], conf
